# ===== notebook cell 2 =====
# ==============================================================
#  GDH-DPQ NEURAL SOLVER  v1
#  Constrained neural construction policy that REPLACES the
#  two-stage Lagrangian dual as the runtime routing solver.
#
#  - Flow-level construction: one decision epoch per satellite-flow,
#    action = choose a gateway (=> biased shortest path) OR DROP.
#  - Feasibility is ARCHITECTURAL: a per-step capacity+lifetime mask
#    makes gateway/ISL overload impossible by construction. DROP is
#    always legal (optional admission), so the mask is never empty.
#  - Label-free training: REINFORCE with a greedy-rollout baseline
#    (self-critical). Reward = -J (your objective) with a large
#    drop penalty, so PLR is minimised first, then delay/hops.
#  - The Lagrangian dual is NOT used to train; it survives ONLY as
#    the offline optimality certificate (gap-to-dual at eval).
#
#  Runs standalone on synthetic instances (Instance.synthetic).
#  To use your real Hypatia instance, call Instance.from_globals(globals())
#  AFTER you have run the benchmark cells (needs V, rho, dist,
#  NBR_LIST, isl_lifetime_s, Cg, Cisl, sat_positions, gateway_positions).
#
#  DESIGN NOTE (v1, deliberate): the learned policy scores GATEWAY
#  selection + DROP (this is the amortisation of the dual price lam_g,
#  and it is fully differentiable via REINFORCE). The PATH given a
#  gateway is the capacity-pruned MIN-PROP-DELAY path (deterministic).
#  A learned per-edge bias (amortised nu_ij) is stubbed but OFF, because
#  routing the bias through argmin-path is non-differentiable and would
#  ship dead parameters. Turn it on only with a differentiable path
#  relaxation or a separate PG signal (future work).
# ==============================================================

import math
import heapq
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
#  0) CONFIG  -- single place for every hyperparameter / constant
# ==============================================================
@dataclass
class CFG:
    # ---- objective weights (match the benchmark) ----
    alpha1: float = 1.0        # delay weight
    alpha2: float = 0.02       # per-hop weight
    ALPHA_HO: float = 0.02     # imminent-handover penalty (alpha3)
    ALPHA4: float = 0.01       # short-lifetime stability penalty
    THO_S: float = 60.0        # handover-imminent threshold (s)
    L_MAX_LIFE: float = 600.0  # stability reference lifetime (s)

    # ---- drop penalty: makes PLR the primary objective ----
    # Cost of dropping 1 Gbps. Must exceed the largest per-Gbps
    # admit cost so "admit when feasible" always dominates.
    KAPPA_DROP: float = 5.0

    # ---- reward shaping (scale-invariant reward) ----
    # reward = served_ratio - W_DELAY*(avgD_ms/100) - W_HOPS*(avg_hops/8)
    # served_ratio (=1-PLR) dominates; delay/hops are small tiebreakers.
    # Bounded ~[-0.1, 1.0] so advantages stay well-conditioned.
    W_DELAY: float = 0.10
    W_HOPS: float = 0.02

    # ---- path-validity filter (v6a) ----
    D_MAX: float = 0.5         # max end-to-end delay (s)
    L_MIN: float = 30.0        # min per-hop ISL lifetime (s)

    # ---- delay constants ----
    c_light: float = 299792.458
    D_TX_ISL: float = 0.0002
    D_PROC_ISL: float = 0.0001
    D_TX_SG: float = 0.0005
    D_PROC_SG: float = 0.0002
    queue_scale: float = 0.002
    QUEUE_MAX: float = 0.05
    queue_eps_frac: float = 1e-3

    # ---- construction controls ----
    K_GW: int = 4              # (v1 legacy) candidate gateways per flow
    MAX_GW_EVAL: int = 16      # (v2) top-M cheapest FEASIBLE gateways scored per flow
                               #   (single Dijkstra reaches all; this caps scoring cost)
    rho_norm: float = 8.0      # rho normaliser (= rho_max)

    # ---- network ----
    hidden: int = 64
    mp_layers: int = 3

    # ---- training ----
    lr: float = 3e-4
    batch: int = 4
    steps: int = 300
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"

    # ---- v2: learned edge-bias head (amortised nu_ij) ----
    # ON by default in v2. The bias enters via a DIFFERENTIABLE cost-along-path
    # feature (path selection is stop-gradient argmin; the cost of the selected
    # path is differentiable in the biases), so the bias head gets a real
    # gradient through the gateway choice's log-prob.
    USE_LEARNED_EDGE_BIAS: bool = True


# ==============================================================
#  1) INSTANCE  -- data contract + self-contained routing helpers
# ==============================================================
@dataclass
class Instance:
    Ns: int
    Ng: int
    V: np.ndarray                    # (Ns, Ng) uint8 visibility
    rho: np.ndarray                  # (Ns,) offered Gbps
    NBR_LIST: list                   # adjacency: list of neighbour lists
    dist: np.ndarray                 # (Ns, Ns) ISL distances (km), 0 if no edge
    isl_lifetime_s: np.ndarray       # (Ns, Ns) remaining ISL lifetime (s)
    Cg: np.ndarray                   # (Ng,) gateway capacities (Gbps)
    Cisl: float                      # per-ISL capacity (Gbps)
    sat_positions: np.ndarray        # (Ns, 3) km
    gateway_positions: np.ndarray    # (Ng, 3) km
    cfg: CFG = field(default_factory=CFG)

    # ---------- static derived ----------
    def __post_init__(self):
        self.isl_edges = [(i, int(j)) for i in range(self.Ns)
                          for j in self.NBR_LIST[i] if int(j) > i]
        self.exit_cands = np.where(self.V.sum(axis=0) > 0)[0]  # gateways w/ >=1 exit sat
        self.deg = np.array([len(n) for n in self.NBR_LIST], dtype=float)

        # ---- precompute for fast rollout (all constant per instance) ----
        c = self.cfg
        base = c.D_TX_ISL + c.D_PROC_ISL
        self.n_edges = len(self.isl_edges)
        self.edge_to_idx = {(i, j): k for k, (i, j) in enumerate(self.isl_edges)}
        # dense edge-index matrix (a,b)->edge id or -1  (avoids tuple/dict in hot loop)
        self.eidx_mat = np.full((self.Ns, self.Ns), -1, dtype=np.int64)
        # per-node adjacency as parallel python lists: (eidx, v, base_weight_s, life_s)
        self.adj = [[] for _ in range(self.Ns)]
        for k, (i, j) in enumerate(self.isl_edges):
            w = float(self.dist[i, j]) / c.c_light + base
            L = float(self.isl_lifetime_s[i, j])
            self.eidx_mat[i, j] = self.eidx_mat[j, i] = k
            self.adj[i].append((k, j, w, L))
            self.adj[j].append((k, i, w, L))
        # per-edge base propagation (prop + tx + proc), indexed by edge id
        self.edge_prop = np.array(
            [float(self.dist[i, j]) / c.c_light + base for (i, j) in self.isl_edges],
            dtype=np.float64) if self.n_edges else np.zeros(0)
        # sat<->gateway propagation matrix (seconds), precomputed once
        diff = self.sat_positions[:, None, :] - self.gateway_positions[None, :, :]
        self.sgprop = (np.linalg.norm(diff, axis=2) / c.c_light).astype(np.float64)  # (Ns,Ng)
        # per-gateway exit-sat arrays + their sg prop (for vectorised candidate scoring)
        self.exit_arr = [np.where(self.V[:, g] > 0)[0].astype(np.int64)
                         for g in range(self.Ng)]
        self.sg_exit = [self.sgprop[self.exit_arr[g], g] if len(self.exit_arr[g]) else
                        np.zeros(0) for g in range(self.Ng)]
        # per-sat list of (gateway, sg_prop) it can see directly (for in-Dijkstra discovery)
        self.sat_gws = [[(g, float(self.sgprop[s, g]))
                         for g in range(self.Ng) if self.V[s, g] > 0]
                        for s in range(self.Ns)]

    # ---------- primitives (mirror the benchmark) ----------
    @staticmethod
    def edge_key(i, j):
        return (int(min(i, j)), int(max(i, j)))

    def isl_prop_s(self, i, j):
        return float(self.dist[i, j] / self.cfg.c_light)

    def sg_prop_s(self, s, g):
        d = float(np.linalg.norm(self.sat_positions[s] - self.gateway_positions[g]))
        return d / self.cfg.c_light

    def sat_gw_km(self, s, g):
        return float(np.linalg.norm(self.sat_positions[s] - self.gateway_positions[g]))

    def exit_sats(self, g):
        return np.where(self.V[:, g] > 0)[0]

    def queue_delay(self, load, cap):
        cap = float(cap)
        leff = min(float(load), (1. - self.cfg.queue_eps_frac) * cap)
        return min(self.cfg.queue_scale / max(cap - leff, 1e-12), self.cfg.QUEUE_MAX)

    def dijkstra(self, start, targets, wfunc):
        pq = [(0., int(start))]
        dv = {int(start): 0.}
        par = {int(start): -1}
        tset = set(int(t) for t in targets)
        hit = None
        while pq:
            du, u = heapq.heappop(pq)
            if du > dv[u] + 1e-15:
                continue
            if u in tset:
                hit = u
                break
            for v in self.NBR_LIST[u]:
                v = int(v)
                w = float(wfunc(u, v))
                if not math.isfinite(w):
                    continue
                nd = du + w
                if v not in dv or nd < dv[v] - 1e-15:
                    dv[v] = nd
                    par[v] = u
                    heapq.heappush(pq, (nd, v))
        if hit is None:
            return None
        path = [int(hit)]
        while par[path[-1]] != -1:
            path.append(int(par[path[-1]]))
        path.reverse()
        return path

    def is_path_valid(self, path, g):
        if path is None or g is None or g < 0:
            return False
        d = 0.0
        for a, b in zip(path[:-1], path[1:]):
            d += self.isl_prop_s(a, b) + self.cfg.D_TX_ISL + self.cfg.D_PROC_ISL
            if float(self.isl_lifetime_s[a, b]) < self.cfg.L_MIN:
                return False
        d += self.sg_prop_s(path[-1], g) + self.cfg.D_TX_SG + self.cfg.D_PROC_SG
        return d <= self.cfg.D_MAX

    def can_admit(self, path, g, flow, gwl, isll):
        if g is None or g < 0:
            return False
        if gwl[g] + flow > self.Cg[g] + 1e-12:
            return False
        for a, b in zip(path[:-1], path[1:]):
            if isll[self.edge_key(a, b)] + flow > self.Cisl + 1e-12:
                return False
        return True

    def admit(self, path, g, flow, gwl, isll):
        gwl[g] += flow
        for a, b in zip(path[:-1], path[1:]):
            isll[self.edge_key(a, b)] += flow

    def full_delay(self, path, g, gwl, isll):
        if path is None or g is None or g < 0:
            return float('nan')
        tot = 0.
        for a, b in zip(path[:-1], path[1:]):
            e = self.edge_key(a, b)
            tot += (self.isl_prop_s(a, b) + self.cfg.D_TX_ISL + self.cfg.D_PROC_ISL
                    + self.queue_delay(float(isll.get(e, 0.)), self.Cisl))
        tot += (self.sg_prop_s(path[-1], g) + self.cfg.D_TX_SG + self.cfg.D_PROC_SG
                + self.queue_delay(float(gwl[g]), float(self.Cg[g])))
        return float(tot)

    def ho_count(self, path):
        return sum(1 for a, b in zip(path[:-1], path[1:])
                   if float(self.isl_lifetime_s[a, b]) <= self.cfg.THO_S)

    def stab_cost(self, path):
        return sum(max(0., self.cfg.L_MAX_LIFE - float(self.isl_lifetime_s[a, b]))
                   for a, b in zip(path[:-1], path[1:]))

    def knn_gateways(self, s, K):
        # K nearest gateways (among those with >=1 exit sat) by sat-gw distance
        cands = self.exit_cands
        if len(cands) == 0:
            return []
        d = np.array([self.sat_gw_km(s, int(g)) for g in cands])
        order = np.argsort(d)[:K]
        return [int(cands[i]) for i in order]

    # ---------- synthetic generator (standalone smoke test) ----------
    @classmethod
    def synthetic(cls, n_planes=5, n_slots=5, n_gw=6, alt_km=550.0,
                  scale=1.0, cfg=None, seed=0, traffic_seed=None,
                  Cg_val=6.0, Cisl_val=3.0, vis_km=2500.0):
        cfg = cfg or CFG()
        rng = np.random.default_rng(seed)                                 # topology
        rng_t = np.random.default_rng(seed if traffic_seed is None else traffic_seed)
        Ns = n_planes * n_slots
        R_earth = 6371.0
        R_orbit = R_earth + alt_km

        def ps_to_i(p, s):
            return p * n_slots + s

        # sat positions on a shell (Walker-ish grid over lon/lat)
        sat_pos = np.zeros((Ns, 3))
        for p in range(n_planes):
            lon = 2 * math.pi * p / n_planes
            for s in range(n_slots):
                lat = math.pi * (s / n_slots) - math.pi / 2
                x = R_orbit * math.cos(lat) * math.cos(lon)
                y = R_orbit * math.cos(lat) * math.sin(lon)
                z = R_orbit * math.sin(lat)
                sat_pos[ps_to_i(p, s)] = [x, y, z]

        # +Grid adjacency
        NBR = [[] for _ in range(Ns)]
        adj = np.zeros((Ns, Ns), dtype=np.uint8)
        for p in range(n_planes):
            for s in range(n_slots):
                i = ps_to_i(p, s)
                for j in (ps_to_i(p, (s + 1) % n_slots),
                          ps_to_i((p + 1) % n_planes, s)):
                    adj[i, j] = adj[j, i] = 1
        for i in range(Ns):
            NBR[i] = [int(j) for j in np.where(adj[i] > 0)[0]]

        dist = np.zeros((Ns, Ns))
        for i in range(Ns):
            for j in NBR[i]:
                if j > i:
                    d = float(np.linalg.norm(sat_pos[i] - sat_pos[j]))
                    dist[i, j] = dist[j, i] = d

        # lifetimes: random remaining life in [15, 1200] s
        life = np.full((Ns, Ns), np.nan)
        for i in range(Ns):
            for j in NBR[i]:
                if j > i:
                    L = float(rng.uniform(15.0, 1200.0))
                    life[i, j] = life[j, i] = L

        # gateways on ground (random lon/lat)
        gw_pos = np.zeros((n_gw, 3))
        for g in range(n_gw):
            lon = rng.uniform(0, 2 * math.pi)
            lat = rng.uniform(-math.pi / 3, math.pi / 3)
            gw_pos[g] = [R_earth * math.cos(lat) * math.cos(lon),
                         R_earth * math.cos(lat) * math.sin(lon),
                         R_earth * math.sin(lat)]

        # visibility: sat sees gw if within a distance threshold (proxy for elevation)
        VIS_KM = vis_km
        V = np.zeros((Ns, n_gw), dtype=np.uint8)
        for i in range(Ns):
            for g in range(n_gw):
                if np.linalg.norm(sat_pos[i] - gw_pos[g]) <= (R_orbit - R_earth) + VIS_KM:
                    V[i, g] = 1
        # guarantee every gateway has >=1 exit sat
        for g in range(n_gw):
            if V[:, g].sum() == 0:
                nearest = int(np.argmin(np.linalg.norm(sat_pos - gw_pos[g], axis=1)))
                V[nearest, g] = 1
        # guarantee every sat sees >=1 gateway (else it can never be served)
        for i in range(Ns):
            if V[i].sum() == 0:
                nearest = int(np.argmin(np.linalg.norm(gw_pos - sat_pos[i], axis=1)))
                V[i, nearest] = 1

        # traffic: lognormal-ish, clipped, scaled  (independent seed)
        rho_raw = rng_t.lognormal(mean=0.0, sigma=0.9, size=Ns)
        rho = np.clip(rho_raw, 0.1, 8.0) * float(scale)

        # capacities (ISL-tight regime)
        Cg = np.full(n_gw, float(Cg_val))
        Cisl = float(Cisl_val)

        return cls(Ns=Ns, Ng=n_gw, V=V, rho=rho, NBR_LIST=NBR, dist=dist,
                   isl_lifetime_s=life, Cg=Cg, Cisl=Cisl,
                   sat_positions=sat_pos, gateway_positions=gw_pos, cfg=cfg)

    # ---------- coordination-gap constructor (decisive test) ----------
    @classmethod
    def coordination_gap(cls, cfg=None, seed=0, traffic_seed=None, scale=1.0,
                         n_cap=8, n_flex=6, C_h=8.0, C_a=30.0,
                         d_cap=1.0, d_flex=2.0):
        """Pure gateway-assignment instance (no ISL edges) with a built-in
        coordination gap. Gateway H (cap C_h) is the NEAREST gateway to
        every sat. Captive sats see ONLY H; flexible sats see H and a far
        gateway A (large cap). rho-descending greedy processes the larger
        flexible flows first, greedily fills H (nearest), then captive sats
        -- which can use only H -- get dropped. Optimal routes flexible->A
        and serves all captive on H. Greedy strictly loses; a forward-looking
        policy can recover it. This isolates whether the policy can learn
        NON-MYOPIC gateway choice on a fixed topology."""
        cfg = cfg or CFG()
        rt = np.random.default_rng(seed if traffic_seed is None else traffic_seed)
        Ns = n_cap + n_flex
        Ng = 2
        # positions: all sats clustered near H; A is far (so H is nearest to all)
        gw_pos = np.array([[6371.0, 0.0, 0.0],      # H
                           [0.0, 6371.0, 0.0]])     # A (far)
        sat_pos = np.zeros((Ns, 3))
        for i in range(Ns):
            sat_pos[i] = [6921.0, rt.uniform(-50, 50), rt.uniform(-50, 50)]
        V = np.zeros((Ns, Ng), dtype=np.uint8)
        V[:n_cap, 0] = 1                 # captive: H only
        V[n_cap:, 0] = 1                 # flexible: H ...
        V[n_cap:, 1] = 1                 # ... and A
        NBR = [[] for _ in range(Ns)]    # NO ISL edges -> pure gateway choice
        dist = np.zeros((Ns, Ns))
        life = np.zeros((Ns, Ns))
        rho = np.zeros(Ns)
        rho[:n_cap] = np.clip(d_cap * rt.uniform(0.8, 1.2, n_cap), 0.1, 8.0) * scale
        rho[n_cap:] = np.clip(d_flex * rt.uniform(0.8, 1.2, n_flex), 0.1, 8.0) * scale
        Cg = np.array([C_h, C_a])
        return cls(Ns=Ns, Ng=Ng, V=V, rho=rho, NBR_LIST=NBR, dist=dist,
                   isl_lifetime_s=life, Cg=Cg, Cisl=99.0,
                   sat_positions=sat_pos, gateway_positions=gw_pos, cfg=cfg)

    # ---------- ISL-coordination-gap constructor (tests edge-bias head) ----------
    @classmethod
    def isl_gap(cls, cfg=None, seed=0, traffic_seed=None, scale=1.0,
                n_cap=6, n_flex=6, d_cap=1.0, d_flex=2.0, Cisl_val=12.0):
        """Single gateway G; the contended resource is an ISL EDGE (J->A), not a
        gateway. Captive sats can reach G only via J->A (short). Flexible sats can
        reach G via J->A (short) OR via a longer J2->M->B detour. rho-descending
        greedy sends the larger flexible flows down the short J->A edge, saturating
        it, so captive flows -- which have no detour -- get dropped. Optimal biases
        the J->A edge so flexible take the detour, freeing J->A for captives.
        Greedy is provably suboptimal; closing it REQUIRES shaping ISL routing,
        i.e. the learned edge bias. Node layout: J=0,A=1,J2=2,M=3,B=4,
        captives=5.., flexibles=.."""
        cfg = cfg or CFG()
        rt = np.random.default_rng(seed if traffic_seed is None else traffic_seed)
        J, A, J2, M, B = 0, 1, 2, 3, 4
        base = 5
        cap_ids = list(range(base, base + n_cap))
        flex_ids = list(range(base + n_cap, base + n_cap + n_flex))
        Ns = base + n_cap + n_flex
        Ng = 1

        edges = {}   # (u,v)->dist_km
        def add(u, v, km):
            edges[(min(u, v), max(u, v))] = km
        for c in cap_ids:
            add(c, J, 500.0)
        for f in flex_ids:
            add(f, J, 500.0); add(f, J2, 500.0)
        add(J, A, 1000.0)                 # BOTTLENECK (short path to G)
        add(J2, M, 1000.0); add(M, B, 1000.0)   # detour (longer)

        dist = np.zeros((Ns, Ns)); life = np.full((Ns, Ns), 1200.0)
        NBR = [[] for _ in range(Ns)]
        for (u, v), km in edges.items():
            dist[u, v] = dist[v, u] = km
            NBR[u].append(v); NBR[v].append(u)

        V = np.zeros((Ns, Ng), dtype=np.uint8)
        V[A, 0] = 1; V[B, 0] = 1          # only A and B see the gateway

        # positions: cluster near origin, gateway near A/B (small, similar sg delay)
        sat_pos = np.zeros((Ns, 3))
        for i in range(Ns):
            sat_pos[i] = [6921.0, rt.uniform(-20, 20), rt.uniform(-20, 20)]
        gw_pos = np.array([[6971.0, 0.0, 0.0]])

        rho = np.zeros(Ns)
        for c in cap_ids:
            rho[c] = np.clip(d_cap * rt.uniform(0.9, 1.1), 0.1, 8.0) * scale
        for f in flex_ids:
            rho[f] = np.clip(d_flex * rt.uniform(0.9, 1.1), 0.1, 8.0) * scale

        return cls(Ns=Ns, Ng=Ng, V=V, rho=rho, NBR_LIST=NBR, dist=dist,
                   isl_lifetime_s=life, Cg=np.array([1e9]), Cisl=float(Cisl_val),
                   sat_positions=sat_pos, gateway_positions=gw_pos, cfg=cfg)

    # ---------- adapter from the benchmark namespace ----------
    @classmethod
    def from_globals(cls, g, cfg=None):
        """Build an Instance from the benchmark's global variables.
        Run this AFTER executing the benchmark cells (sections A-H)."""
        cfg = cfg or CFG()
        Ns = int(g["Ns"]); Ng = int(g["Ng"])
        Cg = np.asarray(g["Cg"], dtype=float)
        if np.ndim(Cg) == 0:
            Cg = np.full(Ng, float(Cg))
        return cls(
            Ns=Ns, Ng=Ng,
            V=np.asarray(g["V_fixed"], dtype=np.uint8),
            rho=np.asarray(g["rho"], dtype=float).copy(),
            NBR_LIST=g["NBR_LIST"],
            dist=np.asarray(g["dist"], dtype=float),
            isl_lifetime_s=np.asarray(g["isl_lifetime_s"], dtype=float),
            Cg=Cg, Cisl=float(g["Cisl"]),
            sat_positions=np.asarray(g["sat_positions"], dtype=float),
            gateway_positions=np.asarray(g["gateway_positions"], dtype=float),
            cfg=cfg,
        )


# ==============================================================
#  2) GRAPH TENSORS  -- built once per instance, fed to the encoder
# ==============================================================
def build_graph_tensors(inst: Instance, device="cpu"):
    cfg = inst.cfg
    Ns, Ng = inst.Ns, inst.Ng
    R = 7000.0  # position normaliser (km)

    # node features: [x,y,z / R, rho/rho_norm, has_vis, deg/4]
    X = np.zeros((Ns, 6), dtype=np.float32)
    X[:, 0:3] = inst.sat_positions / R
    X[:, 3] = inst.rho / cfg.rho_norm
    X[:, 4] = (inst.V.sum(axis=1) > 0).astype(np.float32)
    X[:, 5] = inst.deg / 4.0

    # directed edges + edge attr: [prop_ms, life_norm, cap_headroom(=1 static)]
    src, dst, eattr = [], [], []
    und_attr = []                                  # per-undirected-edge attr (for bias head)
    edge_to_idx = {}                               # edge_key -> row in bias_vec
    for k, (i, j) in enumerate(inst.isl_edges):
        prop = inst.isl_prop_s(i, j)
        life = float(inst.isl_lifetime_s[i, j]) / 1200.0
        edge_to_idx[inst.edge_key(i, j)] = k
        und_attr.append([prop * 1000.0, life, 1.0])
        for a, b in ((i, j), (j, i)):
            src.append(a); dst.append(b)
            eattr.append([prop * 1000.0, life, 1.0])
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device) \
        if src else torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.tensor(np.asarray(eattr, dtype=np.float32), device=device) \
        if eattr else torch.zeros((0, 3), dtype=torch.float32, device=device)
    # undirected endpoints + attr (aligned with inst.isl_edges / edge_to_idx)
    if inst.isl_edges:
        und_u = torch.tensor([i for (i, j) in inst.isl_edges], dtype=torch.long, device=device)
        und_v = torch.tensor([j for (i, j) in inst.isl_edges], dtype=torch.long, device=device)
        und_attr = torch.tensor(np.asarray(und_attr, dtype=np.float32), device=device)
    else:
        und_u = torch.zeros(0, dtype=torch.long, device=device)
        und_v = torch.zeros(0, dtype=torch.long, device=device)
        und_attr = torch.zeros((0, 3), dtype=torch.float32, device=device)

    # gateway visibility pooling weights (row-normalised V^T) and static feats
    Wvis = inst.V.T.astype(np.float32)                      # (Ng, Ns)
    Wvis = Wvis / np.maximum(Wvis.sum(axis=1, keepdims=True), 1.0)
    Wvis = torch.tensor(Wvis, device=device)
    gw_static = np.zeros((Ng, 5), dtype=np.float32)
    gw_static[:, 0:3] = inst.gateway_positions / R
    gw_static[:, 3] = inst.Cg / max(float(inst.Cg.max()), 1e-6)
    gw_static[:, 4] = inst.V.sum(axis=0) / max(Ns, 1)
    gw_static = torch.tensor(gw_static, device=device)

    X = torch.tensor(X, device=device)
    return dict(X=X, edge_index=edge_index, edge_attr=edge_attr,
                Wvis=Wvis, gw_static=gw_static,
                und_u=und_u, und_v=und_v, und_attr=und_attr,
                edge_to_idx=edge_to_idx)


# ==============================================================
#  3) ENCODER  -- edge-conditioned MP + per-edge bias head (nu_ij)
# ==============================================================
class EdgeMPLayer(nn.Module):
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden + edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        self.upd = nn.Linear(2 * hidden, hidden)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.size(1) == 0:
            return F.relu(self.upd(torch.cat([h, torch.zeros_like(h)], dim=-1)))
        src, dst = edge_index[0], edge_index[1]
        m = self.msg(torch.cat([h[src], edge_attr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        deg = torch.zeros(h.size(0), 1, device=h.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device))
        agg = agg / deg.clamp(min=1.0)
        return F.relu(self.upd(torch.cat([h, agg], dim=-1)))


class Encoder(nn.Module):
    def __init__(self, cfg: CFG, node_dim=6, edge_dim=3, gw_static_dim=5):
        super().__init__()
        H = cfg.hidden
        self.cfg = cfg
        self.node_in = nn.Linear(node_dim, H)
        self.layers = nn.ModuleList(
            [EdgeMPLayer(H, edge_dim) for _ in range(cfg.mp_layers)])
        self.gw_proj = nn.Sequential(
            nn.Linear(H + gw_static_dim, H), nn.ReLU(), nn.Linear(H, H))
        # per-edge bias head: [h_u, h_v, edge_attr] -> softplus scalar (>=0)
        self.bias_head = nn.Sequential(
            nn.Linear(2 * H + edge_dim, H), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, gt):
        h = F.relu(self.node_in(gt["X"]))
        for layer in self.layers:
            h = layer(h, gt["edge_index"], gt["edge_attr"])
        gw_pool = gt["Wvis"] @ h
        gw_emb = self.gw_proj(torch.cat([gw_pool, gt["gw_static"]], dim=-1))
        # edge bias (amortised nu_ij), one per undirected edge; in SECONDS scale
        if self.cfg.USE_LEARNED_EDGE_BIAS and gt["und_u"].numel() > 0:
            eu, ev = gt["und_u"], gt["und_v"]
            b_in = torch.cat([h[eu], h[ev], gt["und_attr"]], dim=-1)
            bias_vec = F.softplus(self.bias_head(b_in)).squeeze(-1) * 1e-3  # ~ms->s
        else:
            bias_vec = torch.zeros(gt["und_u"].numel(), device=h.device)
        return h, gw_emb, bias_vec


# ==============================================================
#  4) DECODER  -- per-flow gateway/DROP scoring
# ==============================================================
class Decoder(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        H = cfg.hidden
        # ctx per candidate gateway: [rho_norm, resid_Cg, path_cost_ms,
        #                             hops_norm, min_edge_headroom]
        self.ctx_dim = 5
        self.gw_score = nn.Sequential(
            nn.Linear(H + H + self.ctx_dim, H), nn.ReLU(),
            nn.Linear(H, 1))
        self.drop_score = nn.Sequential(
            nn.Linear(H + 1, H), nn.ReLU(), nn.Linear(H, 1))

    def gateway_logit(self, h_s, gw_emb_g, ctx):
        return self.gw_score(torch.cat([h_s, gw_emb_g, ctx], dim=-1)).squeeze(-1)

    def drop_logit(self, h_s, rho_norm):
        return self.drop_score(torch.cat([h_s, rho_norm.view(1)], dim=-1)).squeeze(-1)


class ConstructionPolicy(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg)
        self.dec = Decoder(cfg)

    def encode(self, inst, device="cpu"):
        gt = build_graph_tensors(inst, device)
        h, gw_emb, bias_vec = self.enc(gt)
        return h, gw_emb, bias_vec, gt["edge_to_idx"]

    # -------- single augmented Dijkstra with in-loop gateway discovery + early stop --------
    def _reach_gateways(self, inst, s, flow, isl_load, bias_list, L_MIN, Cisl,
                        sgpost, gwl, cap_count):
        """One Dijkstra from s over the capacity+lifetime-pruned graph. Discovers
        gateway costs as exit sats are finalised (increasing distance => first
        reached are cheapest) and STOPS once cap_count feasible gateways are found.
        Returns par (list) and best = {g: (cost, exit_sat)}."""
        INF = math.inf
        Ns = inst.Ns
        dist = [INF] * Ns
        par = [-1] * Ns
        dist[s] = 0.0
        pq = [(0.0, s)]
        adj = inst.adj
        sat_gws = inst.sat_gws
        Cg = inst.Cg
        cap = Cisl + 1e-12
        best = {}
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            # discover gateways visible from u (u is an exit sat)
            for (g, sgp) in sat_gws[u]:
                if gwl[g] + flow > Cg[g] + 1e-12:
                    continue
                c = d + sgp + sgpost
                cur = best.get(g)
                if cur is None or c < cur[0]:
                    best[g] = (c, u)
            if len(best) >= cap_count:
                break
            for (eidx, v, bw, life) in adj[u]:
                if isl_load[eidx] + flow > cap:
                    continue
                if life < L_MIN:
                    continue
                nd = d + bw + (bias_list[eidx] if bias_list is not None else 0.0)
                if nd < dist[v]:
                    dist[v] = nd
                    par[v] = u
                    heapq.heappush(pq, (nd, v))
        return par, best

    @staticmethod
    def _path_to(par, node):
        p = [int(node)]
        while par[p[-1]] != -1:
            p.append(int(par[p[-1]]))
        p.reverse()
        return p

    # -------- one construction rollout --------
    def rollout(self, inst: Instance, h, gw_emb, bias_vec, e2i, greedy=False):
        """Constructs a feasible routing solution scoring ALL feasible gateways.
        Returns: mean_logp (tensor), mean_ent (tensor), solution, gwl, isll."""
        cfg = self.cfg
        device = h.device
        gwl = np.zeros(inst.Ng)
        isl_load = np.zeros(inst.n_edges)          # flat per-edge load (fast)
        eidx_mat = inst.eidx_mat
        solution = {}
        logps, ents = [], []
        bias_list = bias_vec.detach().cpu().tolist() if bias_vec.numel() else None
        L_MIN, Cisl = cfg.L_MIN, inst.Cisl
        sgpost = cfg.D_TX_SG + cfg.D_PROC_SG

        order = np.argsort(-inst.rho)
        cap_count = cfg.MAX_GW_EVAL + 8            # discover a small margin past what we score
        for s in order:
            s = int(s); flow = float(inst.rho[s])
            rho_n = torch.tensor(flow / cfg.rho_norm, dtype=torch.float32, device=device)

            par, best = self._reach_gateways(
                inst, s, flow, isl_load, bias_list, L_MIN, Cisl, sgpost, gwl, cap_count)
            if not best:
                continue                          # nothing feasible -> forced drop

            # top-M cheapest discovered gateways
            cand = sorted(((c, g, es) for g, (c, es) in best.items()),
                          key=lambda x: x[0])[:cfg.MAX_GW_EVAL]

            # build candidate paths + differentiable cost feature
            cg, cctx = [], []
            for best_c, g, es in cand:
                path = self._path_to(par, es)
                eids = [int(eidx_mat[a, b]) for a, b in zip(path[:-1], path[1:])]
                hops = len(eids)
                prop_const = float(inst.edge_prop[eids].sum()) if hops else 0.0
                prop_const += float(inst.sgprop[path[-1], g])
                if prop_const + sgpost > cfg.D_MAX:
                    continue
                if bias_list is not None and hops > 0:
                    bias_sum = bias_vec[eids].sum()
                else:
                    bias_sum = torch.zeros((), device=device)
                cost_ms = (float(prop_const) * 1000.0) + bias_sum * 1000.0
                cost_ms = cost_ms.to(torch.float32)
                if hops > 0:
                    min_head = float(min(Cisl - isl_load[e] for e in eids)) / Cisl
                else:
                    min_head = 1.0
                f32 = lambda x: torch.tensor(float(x), dtype=torch.float32, device=device)
                ctx = torch.stack([
                    f32(flow / cfg.rho_norm),
                    f32(max(inst.Cg[g] - gwl[g], 0.) / float(inst.Cg[g])),
                    cost_ms,
                    f32(hops / 8.0),
                    f32(float(min_head)),
                ])
                cg.append((g, path, eids)); cctx.append(ctx)

            if not cg:
                continue

            logits = [self.dec.gateway_logit(h[s], gw_emb[gg], ctx)
                      for (gg, _, _), ctx in zip(cg, cctx)]
            logits.append(self.dec.drop_logit(h[s], rho_n))
            logits = torch.stack(logits)

            probs = F.softmax(logits, dim=-1)
            logp_all = F.log_softmax(logits, dim=-1)
            ents.append(-(probs * logp_all).sum())

            a = int(torch.argmax(probs).item()) if greedy \
                else int(torch.multinomial(probs, 1).item())
            logps.append(logp_all[a])

            if a == len(cg):
                continue                                     # DROP
            g, path, eids = cg[a]
            gwl[g] += flow
            for e in eids:
                isl_load[e] += flow
            solution[s] = (g, path)

        # build isll dict for compute_reward (once, cheap)
        isll = {inst.isl_edges[k]: float(isl_load[k])
                for k in range(inst.n_edges) if isl_load[k] > 0.0}
        isll = defaultdict(float, isll)
        mean_logp = torch.stack(logps).mean() if logps else torch.zeros((), device=device)
        mean_ent = torch.stack(ents).mean() if ents else torch.zeros((), device=device)
        return mean_logp, mean_ent, solution, gwl, isll


# ==============================================================
#  5) REWARD  -- terminal, = -J with drop penalty
# ==============================================================
def compute_reward(inst: Instance, solution, gwl, isll):
    """Returns (reward_float, stats_dict). Reward = -J. Higher is better."""
    cfg = inst.cfg
    J = 0.0
    served = 0.0
    delays, hops = [], []
    for s in range(inst.Ns):
        flow = float(inst.rho[s])
        if s in solution:
            g, path = solution[s]
            d = inst.full_delay(path, g, gwl, isll)
            k = max(len(path) - 1, 0)
            J += (cfg.alpha1 * d + cfg.alpha2 * k
                  + cfg.ALPHA_HO * inst.ho_count(path)
                  + cfg.ALPHA4 * inst.stab_cost(path))
            served += flow
            delays.append(d); hops.append(k)
        else:
            J += cfg.KAPPA_DROP * flow                        # drop penalty
    offered = float(inst.rho.sum())
    # feasibility (must be True by construction)
    mgw = float(np.max(np.maximum(gwl - inst.Cg, 0.))) if inst.Ng else 0.
    mis = max([float(l - inst.Cisl) for l in isll.values()] + [0.])
    feas = (mgw <= 1e-9) and (mis <= 1e-9)
    sr = served / max(offered, 1e-12)
    avgD_ms = float(np.mean(delays)) * 1e3 if delays else 0.0
    avg_hops = float(np.mean(hops)) if hops else 0.0
    # Scale-invariant reward: served ratio dominates (PLR primary),
    # delay/hops are small tiebreakers. Bounded ~[-0.1, 1.0].
    reward = sr - cfg.W_DELAY * (avgD_ms / 100.0) - cfg.W_HOPS * (avg_hops / 8.0)
    stats = dict(
        served_ratio=sr, plr=1.0 - sr,
        avgD_ms=(float(np.mean(delays)) * 1e3 if delays else float('nan')),
        avg_hops=avg_hops, feas=feas, J=J,
    )
    return reward, stats


# ==============================================================
#  6) TRAIN  -- REINFORCE with greedy-rollout baseline (self-critical)
# ==============================================================
def make_instance(cfg, seed, scale=None):
    if scale is None:
        scale = float(np.random.default_rng(seed).uniform(0.5, 1.5))
    return Instance.synthetic(cfg=cfg, seed=seed, scale=scale)


def train(policy, cfg: CFG, instance_fn=None, log_every=20,
          val_insts=None, val_every=10):
    """instance_fn(step_seed) -> Instance. Defaults to synthetic.
    val_insts: optional list of fixed Instances; if given, greedy-eval on them
    every val_every steps and RESTORE the best-scoring checkpoint at the end
    (defends against the noisy last step). Uses cosine LR decay."""
    device = cfg.device
    policy.to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    if instance_fn is None:
        instance_fn = lambda seed: make_instance(cfg, seed)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.steps, 1), eta_min=cfg.lr * 0.05)

    history = []
    rng = np.random.default_rng(cfg.seed)
    best_val, best_state = -1e9, None
    import copy
    for step in range(cfg.steps):
        opt.zero_grad()
        advs, logp_terms, ent_terms = [], [], []
        batch_R, batch_plr, batch_feas = [], [], []
        for b in range(cfg.batch):
            seed = int(rng.integers(0, 1 << 30))
            inst = instance_fn(seed)
            h, gw_emb, bias_vec, e2i = policy.encode(inst, device)

            # sampled rollout (grad)
            logp, ent, sol, gwl, isll = policy.rollout(
                inst, h, gw_emb, bias_vec, e2i, greedy=False)
            R_sample, st = compute_reward(inst, sol, gwl, isll)

            # greedy baseline (no grad) -- fresh instance copy state
            with torch.no_grad():
                inst_b = instance_fn(seed)  # same instance (same seed)
                hb, gwb, bvb, e2ib = policy.encode(inst_b, device)
                _, _, sol_g, gwl_g, isll_g = policy.rollout(
                    inst_b, hb, gwb, bvb, e2ib, greedy=True)
                R_greedy, _ = compute_reward(inst_b, sol_g, gwl_g, isll_g)

            adv = R_sample - R_greedy                     # self-critical baseline
            advs.append(adv); logp_terms.append(logp); ent_terms.append(ent)
            batch_R.append(R_sample); batch_plr.append(st["plr"]); batch_feas.append(st["feas"])

        # standardize advantages across the batch (variance reduction) so the
        # gradient scale is decoupled from raw reward magnitude
        adv_t = torch.tensor(advs, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        batch_loss = torch.stack(
            [-a.detach() * lp - cfg.entropy_coef * e
             for a, lp, e in zip(adv_t, logp_terms, ent_terms)]).mean()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        # keep-best-by-validation: greedy-eval on fixed val instances, snapshot best
        val_sr = None
        if val_insts is not None and (step % val_every == 0 or step == cfg.steps - 1):
            srs = [evaluate(policy, vi)["served_ratio"] for vi in val_insts]
            val_sr = float(np.mean(srs))
            if val_sr > best_val:
                best_val = val_sr
                best_state = copy.deepcopy(policy.state_dict())

        rec = dict(step=step, loss=float(batch_loss.item()),
                   R=float(np.mean(batch_R)), plr=float(np.mean(batch_plr)),
                   feas=float(np.mean(batch_feas)), val_sr=val_sr)
        history.append(rec)
        if step % log_every == 0 or step == cfg.steps - 1:
            vtxt = f"  val_SR={val_sr:.3f}" if val_sr is not None else ""
            print(f"[{step:4d}] loss={rec['loss']:+.4f}  R={rec['R']:+.3f}  "
                  f"PLR={rec['plr']:.4f}  feas={rec['feas']:.2f}{vtxt}")

    # restore best-validation checkpoint (not the noisy last step)
    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"[train] restored best-val checkpoint: val_SR={best_val:.3f}")
    return history


# ==============================================================
#  7) EVAL  -- greedy decode, report KPIs (+ optional gap-to-dual)
# ==============================================================
def greedy_nearest_baseline(inst: Instance):
    """Non-learned reference: each flow (rho-desc) -> nearest feasible
    gateway via min-prop path, admit if it fits, else drop. This is the
    'obvious heuristic' the learned policy must beat to justify itself."""
    gwl = np.zeros(inst.Ng); isll = defaultdict(float); sol = {}
    for s in np.argsort(-inst.rho):
        s = int(s); flow = float(inst.rho[s])
        for g in inst.knn_gateways(s, inst.cfg.K_GW):
            if gwl[g] + flow > inst.Cg[g] + 1e-12:
                continue
            def wf(u, v, _g=g):
                if isll[inst.edge_key(u, v)] + flow > inst.Cisl + 1e-12:
                    return math.inf
                if float(inst.isl_lifetime_s[u, v]) < inst.cfg.L_MIN:
                    return math.inf
                return inst.isl_prop_s(u, v)
            path = inst.dijkstra(s, inst.exit_sats(g), wf)
            if path is None or not inst.can_admit(path, g, flow, gwl, isll):
                continue
            if not inst.is_path_valid(path, g):
                continue
            inst.admit(path, g, flow, gwl, isll); sol[s] = (g, path); break
    _, st = compute_reward(inst, sol, gwl, isll)
    return st


def paired_eval(policy, build_fn, scale, seeds):
    """Paired comparison over many traffic realizations at a fixed scale.
    build_fn(traffic_seed, scale) -> Instance (same topology, varied load).
    Returns per-scale means/stds and the PAIRED difference (policy - greedy),
    which is what tells you if any separation is real vs noise."""
    sp, sg, diff = [], [], []
    for ts in seeds:
        rp = evaluate(policy, build_fn(ts, scale))
        rg = greedy_nearest_baseline(build_fn(ts, scale))
        sp.append(rp["served_ratio"]); sg.append(rg["served_ratio"])
        diff.append(rp["served_ratio"] - rg["served_ratio"])
    sp, sg, diff = np.array(sp), np.array(sg), np.array(diff)
    n = len(seeds)
    se = diff.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    tstat = diff.mean() / se if se > 0 else 0.0
    return dict(scale=scale, n=n,
                policy_mean=sp.mean(), policy_std=sp.std(),
                greedy_mean=sg.mean(), greedy_std=sg.std(),
                diff_mean=diff.mean(), diff_std=diff.std(),
                diff_se=se, tstat=tstat)


@torch.no_grad()
def evaluate(policy, inst: Instance, dual_served_ratio=None):
    policy.eval()
    h, gw_emb, bias_vec, e2i = policy.encode(inst, policy.cfg.device)
    _, _, sol, gwl, isll = policy.rollout(
        inst, h, gw_emb, bias_vec, e2i, greedy=True)
    R, st = compute_reward(inst, sol, gwl, isll)
    out = dict(reward=R, **st, routed=len(sol), offered=float(inst.rho.sum()))
    if dual_served_ratio is not None:
        out["gap_to_dual_pp"] = (dual_served_ratio - st["served_ratio"]) * 100.0
    policy.train()
    return out


# ==============================================================
#  8) QUICKSTART  (Jupyter: just run these lines)
# ==============================================================
def compare_on_real(g, cfg=None, scales=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                    train_steps=400, train_band=(0.6, 1.5),
                    references=(("CA-DSP", "run_cadsp"),
                                ("GDH-DPQ", "run_gdh_dpq")),
                    log=True):
    """Run the WHOLE experiment on your benchmark instance.
    Call as: rows, policy = compare_on_real(globals())  -- AFTER running the
    benchmark cells (needs V_fixed, rho/rho_base, Cg, Cisl, dist, NBR_LIST,
    isl_lifetime_s, sat_positions, gateway_positions, Ns, Ng, and the scheme
    functions run_cadsp / run_gdh_dpq in scope).

    Trains the policy on your fixed topology (traffic varied in train_band),
    then at each scale reports policy served-ratio next to each reference
    scheme, plus the gap the policy leaves to the best reference. The
    reference schemes read the module-level `rho`, which this function sets
    per scale and restores afterwards.
    """
    cfg = cfg or CFG(steps=train_steps, batch=8)
    rho_base = np.asarray(g.get("rho_base", g["rho"]), dtype=float).copy()
    V = g["V_fixed"]

    # ---- train on the real fixed topology, traffic varied ----
    def train_inst(seed):
        r = np.random.default_rng(seed)
        inst = Instance.from_globals(g, cfg=cfg)
        inst.rho = np.clip(rho_base * r.uniform(*train_band), 0.1, None)
        return inst

    # fixed validation instances across the overload band (for keep-best)
    val_scales = [s for s in (1.0, 1.25, 1.5) if train_band[0] <= s <= train_band[1] + 1e-9]
    if not val_scales:
        val_scales = [1.0]
    val_insts = []
    for vs in val_scales:
        vi = Instance.from_globals(g, cfg=cfg)
        vi.rho = rho_base * float(vs)
        val_insts.append(vi)

    policy = ConstructionPolicy(cfg)
    if log:
        print(f"=== training policy on real instance "
              f"(Ns={g['Ns']}, Ng={g['Ng']}, {cfg.steps} steps, "
              f"val@{val_scales}) ===")
    train(policy, cfg, instance_fn=train_inst,
          log_every=max(1, cfg.steps // 12),
          val_insts=val_insts, val_every=max(1, cfg.steps // 20))

    # ---- compare across scales ----
    rho_saved = np.asarray(g["rho"], dtype=float).copy()
    rows = []
    for sc in scales:
        inst = Instance.from_globals(g, cfg=cfg)
        inst.rho = rho_base * float(sc)
        rp = evaluate(policy, inst)
        row = {"scale": float(sc),
               "policy_SR": rp["served_ratio"],
               "policy_avgD_ms": rp["avgD_ms"],
               "policy_feas": rp["feas"]}
        g["rho"] = rho_base * float(sc)                 # scheme fns read this
        for name, fn in references:
            sr = float("nan")
            if fn in g and callable(g[fn]):
                try:
                    out = g[fn](V)
                    sr = float(out.get("served_ratio", float("nan")))
                except Exception as e:
                    if log:
                        print(f"  [{name}] failed at scale {sc}: {e}")
            row[f"{name}_SR"] = sr
        rows.append(row)
    g["rho"] = rho_saved                                # restore

    # ---- print table ----
    if log:
        ref_names = [n for n, _ in references]
        hdr = f"\n{'scale':>6} | {'policy SR':>9} | " + \
              " | ".join(f"{n+' SR':>10}" for n in ref_names) + \
              " | gap_to_best_ref"
        print(hdr)
        for r in rows:
            refs = [r[f"{n}_SR"] for n in ref_names]
            best = np.nanmax(refs) if any(not math.isnan(x) for x in refs) else float("nan")
            gap = (r["policy_SR"] - best) * 100 if not math.isnan(best) else float("nan")
            cells = " | ".join(f"{r[f'{n}_SR']:10.3f}" for n in ref_names)
            print(f"{r['scale']:6.2f} | {r['policy_SR']:9.3f} | {cells} | "
                  f"{gap:+.1f}pp")
        print("\ngap_to_best_ref > 0  => policy beats every reference at that scale.")
        print("policy near GDH-DPQ (small gap) at heuristic speed = the result.")
    return rows, policy


# ==============================================================
#  9) LEVEL-1 GENERALIZATION  -- across orbital time snapshots
# ==============================================================
def build_snapshot(g, t_offset_s, cfg=None, rebuild_lifetimes=False):
    """Construct an Instance at prop_time + t_offset_s using the benchmark's own
    propagation functions in the namespace g. Re-propagates satellite POSITIONS
    (=> new ISL distances) and rebuilds gateway VISIBILITY. Traffic (rho_base) and
    ISL lifetimes are held fixed by default (they evolve slowly; set
    rebuild_lifetimes=True to recompute lifetimes too -- slower). Requires in g:
    prop_time, satrecs, adj, prop_positions, build_visibility, ELEV_FIXED,
    isl_lifetime_s, gateway_positions, Ns, Ng, NBR_LIST, Cg, Cisl, rho_base."""
    from datetime import timedelta
    cfg = cfg or CFG()
    Ns, Ng = int(g["Ns"]), int(g["Ng"])
    T = g["prop_time"] + timedelta(seconds=float(t_offset_s))

    pos, good = g["prop_positions"](T, g["satrecs"])
    # patch any bad sats with neighbour mean (mirrors the benchmark)
    if not np.all(good):
        NBR = g["NBR_LIST"]
        for i in np.where(~good)[0]:
            nb = [n for n in NBR[i] if good[n]]
            if nb:
                pos[i] = np.mean([pos[n] for n in nb], axis=0); good[i] = True

    # recompute ISL distances on the fixed +Grid edges
    adj = g["adj"]
    dist = np.zeros((Ns, Ns))
    upper = np.argwhere(np.triu(adj, 1) > 0)
    for i, j in upper:
        d = float(np.linalg.norm(pos[i] - pos[j]))
        dist[i, j] = dist[j, i] = d

    V = g["build_visibility"](pos, g["ELEV_FIXED"])
    if rebuild_lifetimes and "build_lifetimes" in g:
        life = g["build_lifetimes"](T, adj, 1.5 * float(dist.max()),
                                    g.get("LIFE_DT_S", 20.0),
                                    g.get("LIFE_HORIZON_S", 1200.0))
    else:
        life = np.asarray(g["isl_lifetime_s"], dtype=float)

    Cg = np.asarray(g["Cg"], dtype=float)
    if np.ndim(Cg) == 0:
        Cg = np.full(Ng, float(Cg))
    rho = np.asarray(g.get("rho_base", g["rho"]), dtype=float).copy()
    return Instance(Ns=Ns, Ng=Ng, V=np.asarray(V, dtype=np.uint8), rho=rho,
                    NBR_LIST=g["NBR_LIST"], dist=dist, isl_lifetime_s=life,
                    Cg=Cg, Cisl=float(g["Cisl"]),
                    sat_positions=np.asarray(pos, dtype=float),
                    gateway_positions=np.asarray(g["gateway_positions"], dtype=float),
                    cfg=cfg)


def run_time_generalization(train_snaps, test_snaps, cfg,
                            band=(0.6, 1.5), eval_scale=1.0, log=True):
    """Train on train_snaps (traffic varied in band), evaluate on HELD-OUT
    test_snaps at eval_scale. Keep-best uses ONLY train snapshots (no leakage
    from the test set). Prints train vs held-out served-ratio."""
    base_rhos = [s.rho.copy() for s in train_snaps]

    def train_inst(seed):
        r = np.random.default_rng(seed)
        k = int(r.integers(len(train_snaps)))
        snap = train_snaps[k]
        snap.rho = np.clip(base_rhos[k] * r.uniform(*band), 0.1, None)
        return snap

    # validation = train snapshots at eval_scale (in-distribution; keeps test clean)
    val_insts = []
    for i, s in enumerate(train_snaps):
        s.rho = base_rhos[i] * eval_scale
        val_insts.append(s)

    policy = ConstructionPolicy(cfg)
    if log:
        print(f"=== TIME-SNAPSHOT generalization: train on {len(train_snaps)} "
              f"snapshots, test on {len(test_snaps)} held-out ===")
    train(policy, cfg, instance_fn=train_inst,
          log_every=max(1, cfg.steps // 12),
          val_insts=val_insts, val_every=max(1, cfg.steps // 20))

    # restore train rho, then eval
    for i, s in enumerate(train_snaps):
        s.rho = base_rhos[i] * eval_scale
    tr = [evaluate(policy, s)["served_ratio"] for s in train_snaps]
    te_rhos = [s.rho.copy() for s in test_snaps]
    for i, s in enumerate(test_snaps):
        s.rho = te_rhos[i] * eval_scale
    te = [evaluate(policy, s)["served_ratio"] for s in test_snaps]

    if log:
        print(f"\n  train snapshots  SR: "
              f"{', '.join(f'{x:.3f}' for x in tr)}  (mean {np.mean(tr):.3f})")
        print(f"  HELD-OUT test SR: "
              f"{', '.join(f'{x:.3f}' for x in te)}  (mean {np.mean(te):.3f})")
        gap = (np.mean(tr) - np.mean(te)) * 100
        print(f"\n  generalization gap (train - test): {gap:+.1f}pp")
        print("  small gap => policy transfers across orbital snapshots "
              "without retraining.")
    return dict(train_sr=tr, test_sr=te, policy=policy)


def time_generalization_from_globals(g, cfg=None,
                                     train_offsets=(0, 120, 240, 360, 480),
                                     test_offsets=(60, 180, 300, 420),
                                     band=(0.6, 1.5), eval_scale=1.0):
    """One-call level-1 test from the benchmark namespace. Builds snapshots at the
    given time offsets (seconds), trains on train_offsets, tests on held-out
    test_offsets. Call: res = time_generalization_from_globals(globals())."""
    cfg = cfg or CFG(steps=250, batch=8, MAX_GW_EVAL=16,
                     USE_LEARNED_EDGE_BIAS=False, lr=3e-4)
    print(f"building {len(train_offsets)} train + {len(test_offsets)} test snapshots ...")
    train_snaps = [build_snapshot(g, off, cfg) for off in train_offsets]
    test_snaps = [build_snapshot(g, off, cfg) for off in test_offsets]
    # report how much visibility shifts across snapshots (sanity: they DIFFER)
    v0 = train_snaps[0].V
    for off, s in zip(test_offsets, test_snaps):
        chg = int(np.abs(s.V.astype(int) - v0.astype(int)).sum())
        print(f"  test snapshot +{off}s: visibility entries changed vs t=0: {chg}")
    return run_time_generalization(train_snaps, test_snaps, cfg,
                                   band=band, eval_scale=eval_scale)


# ==============================================================
#  10) LEVEL-2 GENERALIZATION  -- across constellation DESIGNS
# ==============================================================
def build_shell(g, incl, alt, n_planes, n_slots, cfg=None,
                incl_tol=0.4, alt_tol=25.0, traffic_seed=0,
                target_util=0.73, rho_clip=(0.1, 8.0)):
    """Build a REAL constellation Instance from one Starlink shell in the CSV.
    Filters df_csv to satellites near (incl, alt), Walker-bins them into
    n_planes x n_slots, propagates to prop_time, builds +Grid ISLs, reuses the
    real gateways, and sets traffic to a clipped-lognormal scaled to target_util
    (so shells are load-comparable -> a clean geometry-only generalization test).
    Reuses benchmark helpers from g: df_csv, prop_time, datetime_to_jd_fr,
    prop_positions, build_visibility, gateway_positions, Ng, Cg, Cisl, ELEV_FIXED.
    ISL lifetimes are set constant by default (secondary edge feature)."""
    import pandas as pd
    from sgp4.api import Satrec, WGS72
    cfg = cfg or CFG()
    df = g["df_csv"]
    prop_time = g["prop_time"]
    dtj = g["datetime_to_jd_fr"]
    Ns = n_planes * n_slots

    # 1) filter to shell (inclination + altitude from mean motion)
    mu = 398600.4418
    n_rad = df["MEAN_MOTION"].to_numpy() * 2 * np.pi / 86400.0
    alt_all = (mu / n_rad ** 2) ** (1 / 3) - 6371.0
    mask = (np.abs(df["INCLINATION"].to_numpy() - incl) <= incl_tol) & \
           (np.abs(alt_all - alt) <= alt_tol)
    sub = df[mask].reset_index(drop=True)
    if len(sub) < Ns:
        raise ValueError(f"shell ({incl},{alt}) has {len(sub)} sats < {Ns} needed; "
                         f"reduce n_planes*n_slots or widen tol")

    # 2) Walker-bin: RAAN rank -> planes, mean anomaly -> slots
    raan = sub["RA_OF_ASC_NODE"].to_numpy() % 360
    ma = sub["MEAN_ANOMALY"].to_numpy() % 360
    order = np.argsort(raan); ranks = np.empty_like(order); ranks[order] = np.arange(len(sub))
    plane_id = np.clip((ranks * n_planes) // len(sub), 0, n_planes - 1)
    targets = np.linspace(0, 360, n_slots, endpoint=False)
    pick = np.full((n_planes, n_slots), -1, int)
    for p in range(n_planes):
        idx = np.where(plane_id == p)[0]
        if len(idx) == 0:
            idx = np.where(plane_id == max(0, p - 1))[0]
        pa = ma[idx]
        for s in range(n_slots):
            diff = np.abs(((pa - targets[s] + 180) % 360) - 180)
            pick[p, s] = int(idx[np.argmin(diff)])
    sel = pick.reshape(-1)

    # 3) SGP4 init the selected satellites
    JD_REF = 2433281.5
    satrecs = []
    for k in range(Ns):
        row = sub.iloc[int(sel[k])]
        mm = float(row["MEAN_MOTION"]); ecc = float(row["ECCENTRICITY"])
        inc = np.deg2rad(float(row["INCLINATION"])); rn = np.deg2rad(float(row["RA_OF_ASC_NODE"]))
        argp = np.deg2rad(float(row["ARG_OF_PERICENTER"])); M = np.deg2rad(float(row["MEAN_ANOMALY"]))
        bstar = float(row["BSTAR"]) if ("BSTAR" in row and pd.notna(row["BSTAR"])) else 0.
        ep = pd.to_datetime(row["EPOCH"], utc=True).to_pydatetime()
        ji, fr = dtj(ep); ep_d = (ji + fr) - JD_REF
        sat = Satrec()
        sat.sgp4init(WGS72, "i",
                     int(row["NORAD_CAT_ID"]) if "NORAD_CAT_ID" in row else 100000 + k,
                     ep_d, bstar, 0., 0., ecc, argp, inc, M, mm * 2 * np.pi / 1440., rn)
        satrecs.append(sat)

    # 4) propagate to prop_time
    pos, good = g["prop_positions"](prop_time, satrecs)

    # 5) +Grid adjacency + distances
    def ps_to_i(p, s): return p * n_slots + s
    adj = np.zeros((Ns, Ns), np.uint8)
    for p in range(n_planes):
        for s in range(n_slots):
            i = ps_to_i(p, s)
            for j in (ps_to_i(p, (s + 1) % n_slots), ps_to_i((p + 1) % n_planes, s)):
                adj[i, j] = adj[j, i] = 1
    NBR = [list(map(int, np.where(adj[i] > 0)[0])) for i in range(Ns)]
    if not np.all(good):
        for i in np.where(~good)[0]:
            nb = [n for n in NBR[i] if good[n]]
            if nb:
                pos[i] = np.mean([pos[n] for n in nb], axis=0); good[i] = True
    dist = np.zeros((Ns, Ns))
    for i in range(Ns):
        for j in NBR[i]:
            if j > i:
                d = float(np.linalg.norm(pos[i] - pos[j])); dist[i, j] = dist[j, i] = d

    # 6) gateways (reuse real ones) + capacities
    gw_pos = np.asarray(g["gateway_positions"], float); Ng = int(g["Ng"])
    Cg = np.asarray(g["Cg"], float)
    if np.ndim(Cg) == 0:
        Cg = np.full(Ng, float(Cg))
    Cisl = float(g["Cisl"])

    # 7) visibility (reuse real elevation-based function) + guarantees
    V = np.asarray(g["build_visibility"](pos, g["ELEV_FIXED"]), np.uint8)
    for gg in range(Ng):
        if V[:, gg].sum() == 0:
            V[int(np.argmin(np.linalg.norm(pos - gw_pos[gg], axis=1))), gg] = 1

    # 8) traffic: clipped lognormal scaled so offered = target_util * total GW cap
    rng = np.random.default_rng(traffic_seed)
    rho = np.clip(rng.lognormal(0.0, 0.9, Ns), *rho_clip)
    total_cap = float(Cg.sum())
    if rho.sum() > 0:
        rho *= (target_util * total_cap) / rho.sum()
    rho = np.clip(rho, rho_clip[0], None)

    # 9) lifetimes (constant; secondary feature)
    life = np.full((Ns, Ns), float(g.get("LIFE_HORIZON_S", 1200.0)))

    return Instance(Ns=Ns, Ng=Ng, V=V, rho=rho, NBR_LIST=NBR, dist=dist,
                    isl_lifetime_s=life, Cg=Cg, Cisl=Cisl,
                    sat_positions=np.asarray(pos, float),
                    gateway_positions=gw_pos, cfg=cfg)


def cross_shell_generalization(g, train_shells, test_shells, cfg=None,
                               band=(0.6, 1.5), eval_scale=1.0):
    """REAL cross-design test across Starlink shells. train_shells / test_shells
    are lists of dicts: {'incl':.., 'alt':.., 'n_planes':.., 'n_slots':..}.
    Trains on the train shells (traffic varied), evaluates on HELD-OUT shells."""
    cfg = cfg or CFG(steps=250, batch=8, MAX_GW_EVAL=16,
                     USE_LEARNED_EDGE_BIAS=False, lr=3e-4)
    print("building real shells ...")
    tr = [build_shell(g, cfg=cfg, **s) for s in train_shells]
    te = [build_shell(g, cfg=cfg, **s) for s in test_shells]
    for s, inst in zip(train_shells, tr):
        print(f"  TRAIN shell incl={s['incl']} alt={s['alt']}: "
              f"Ns={inst.Ns} visible={(inst.V.sum(1) > 0).sum()} offered={inst.rho.sum():.0f}")
    for s, inst in zip(test_shells, te):
        print(f"  TEST  shell incl={s['incl']} alt={s['alt']}: "
              f"Ns={inst.Ns} visible={(inst.V.sum(1) > 0).sum()} offered={inst.rho.sum():.0f}")
    return run_time_generalization(tr, te, cfg, band=band, eval_scale=eval_scale)


def finetune_on_shell(policy, target_inst, cfg=None, steps=40, lr=2e-4,
                      batch=6, band=(0.7, 1.3), eval_scale=1.0, eval_every=10,
                      log=True):
    """FEW-SHOT ADAPTATION (Path C). Start from a pretrained `policy` and adapt it
    to a single target constellation with a few REINFORCE steps. Returns the
    served-ratio trajectory [(0, zero_shot), (eval_every, ...), ...] so you can
    see SR climb from zero-shot to adapted. Reframes cross-design as
    'pretrain once, adapt to a new constellation in minutes' rather than
    zero-shot transfer (which fails)."""
    cfg = cfg or policy.cfg
    device = cfg.device
    base_rho = target_inst.rho.copy()

    def ev():
        target_inst.rho = base_rho * eval_scale
        return evaluate(policy, target_inst)["served_ratio"]

    traj = [(0, ev())]
    if log:
        print(f"  zero-shot SR (no adaptation): {traj[0][1]:.3f}")
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    best_sr, best_state = traj[0][1], None
    import copy
    for step in range(1, steps + 1):
        opt.zero_grad()
        advs, logps, ents = [], [], []
        for b in range(batch):
            rr = float(np.random.default_rng(int(rng.integers(0, 1 << 30))).uniform(*band))
            target_inst.rho = np.clip(base_rho * rr, 0.1, None)
            h, gw, bv, e2i = policy.encode(target_inst, device)
            lp, en, sol, gwl, isll = policy.rollout(target_inst, h, gw, bv, e2i, greedy=False)
            Rs, _ = compute_reward(target_inst, sol, gwl, isll)
            with torch.no_grad():
                _, _, solg, gwlg, isllg = policy.rollout(target_inst, h, gw, bv, e2i, greedy=True)
                Rg, _ = compute_reward(target_inst, solg, gwlg, isllg)
            advs.append(Rs - Rg); logps.append(lp); ents.append(en)
        adv_t = torch.tensor(advs, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        loss = torch.stack([-a.detach() * lp - cfg.entropy_coef * e
                            for a, lp, e in zip(adv_t, logps, ents)]).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        if step % eval_every == 0 or step == steps:
            sr = ev(); traj.append((step, sr))
            if sr > best_sr:
                best_sr = sr; best_state = copy.deepcopy(policy.state_dict())
            if log:
                print(f"  after {step:3d} adapt steps: SR={sr:.3f}")
    if best_state is not None:
        policy.load_state_dict(best_state)
    if log:
        print(f"  => zero-shot {traj[0][1]:.3f}  ->  adapted {best_sr:.3f}  "
              f"(+{(best_sr - traj[0][1]) * 100:.1f}pp in {steps} steps)")
    return traj


def assemble_results(policy_rows, g, plr_thresh=0.05, df_name="df_ng",
                     schemes=("Direct Drop", "DSP", "CA-DSP", "DBPR", "GPSR", "Proposed"),
                     log=True):
    """Combine the AI policy sweep with the benchmark's reference schemes (df_ng)
    into publication-ready tables:
      (1) PLR vs scale for ALL schemes + AI,
      (2) delay & hops comparison ONLY in the feasible regime (AI PLR < plr_thresh)
          -- survivorship-safe, apples-to-apples,
      (3) MATLAB overlay arrays for the AI (baseline arrays already in df_ng).
    Reads df_ng from g if present; else reports the AI alone."""
    df = g.get(df_name, None)
    prow = {round(r["scale"], 4): r for r in policy_rows}
    scales = sorted(prow.keys())

    def ref(scheme, scale, col):
        if df is None:
            return float("nan")
        m = df[(df["scheme"] == scheme) & (np.isclose(df["scale_active"], scale))]
        return float(m[col].iloc[0]) if len(m) else float("nan")

    if log:
        # (1) PLR vs scale, all schemes + AI
        cols = list(schemes) + ["AI"]
        print("=== (1) PLR vs traffic scale ===")
        print("scale | " + " | ".join(f"{c:>11}" for c in cols))
        for s in scales:
            cells = [f"{ref(sch, s, 'packet_loss_ratio'):11.4f}" for sch in schemes]
            cells.append(f"{prow[s]['plr']:11.4f}")
            print(f"{s:5.2f} | " + " | ".join(cells))

        # (2) delay + hops ONLY where AI is feasible (PLR < thresh)
        feas = [s for s in scales if prow[s]["plr"] < plr_thresh]
        print(f"\n=== (2) delay (ms) & hops -- FEASIBLE regime only "
              f"(AI PLR < {plr_thresh:.0%}): scales {feas} ===")
        print("       delay is survivorship-biased outside this regime; compare here only.")
        print(f"\n{'scale':>5} | {'metric':>6} | " +
              " | ".join(f"{sch:>9}" for sch in ('Proposed', 'CA-DSP', 'DSP')) +
              f" | {'AI':>9}")
        for s in feas:
            dline = [ref('Proposed', s, 'avgD_s') * 1e3, ref('CA-DSP', s, 'avgD_s') * 1e3,
                     ref('DSP', s, 'avgD_s') * 1e3, prow[s]['avgD_ms']]
            hline = [ref('Proposed', s, 'avg_hops'), ref('CA-DSP', s, 'avg_hops'),
                     ref('DSP', s, 'avg_hops'), prow[s]['hops']]
            print(f"{s:5.2f} | {'delay':>6} | " + " | ".join(f"{x:9.1f}" for x in dline))
            print(f"{'':>5} | {'hops':>6} | " + " | ".join(f"{x:9.2f}" for x in hline))

        # (3) AI overlay arrays
        print(f"\n=== (3) AI overlay arrays ===")
        def jarr(key, sc=1.0):
            return " ".join("{:.6f}".format(prow[s][key] * sc) for s in scales)
        sx = " ".join(f"{s:.2f}" for s in scales)
        print(f"ai_tr_x_scale = [{sx}];")
        print(f"ai_tr_plr  = [{jarr('plr')}];")
        print(f"ai_tr_sr   = [{jarr('sr')}];")
        print(f"ai_tr_avgD = [{jarr('avgD_ms', 1e-3)}];")
        print(f"ai_tr_hops = [{jarr('hops')}];")

    return dict(scales=scales, policy=prow,
                feasible_scales=[s for s in scales if prow[s]["plr"] < plr_thresh])


def dual_on_shell(g, shell_kwargs, scale=1.0, cfg=None, fn="run_gdh_dpq"):
    """Run the benchmark's dual (run_gdh_dpq) on a build_shell instance to get a
    dual served-ratio reference on a held-out shell. Sets the global rho/V the
    scheme function reads, calls it, restores. Returns served_ratio."""
    inst = build_shell(g, cfg=cfg, **shell_kwargs)
    inst.rho = inst.rho * float(scale)
    # the scheme functions read module globals; temporarily install shell state
    saved = {k: g.get(k) for k in ("rho", "V_fixed", "Ns", "Ng", "dist",
                                   "NBR_LIST", "isl_lifetime_s", "sat_positions")}
    g["rho"] = inst.rho; g["V_fixed"] = inst.V; g["Ns"] = inst.Ns; g["Ng"] = inst.Ng
    g["dist"] = inst.dist; g["NBR_LIST"] = inst.NBR_LIST
    g["isl_lifetime_s"] = inst.isl_lifetime_s; g["sat_positions"] = inst.sat_positions
    try:
        out = g[fn](inst.V)
        sr = float(out.get("served_ratio", float("nan")))
    finally:
        for k, v in saved.items():
            if v is not None:
                g[k] = v
    return sr


def timing_comparison(policy, g, n_repeats=8, run_dual=False, cfg=None):
    """Measure inference cost: pure GNN forward pass (encode), full AI routing
    table (encode + rollout, incl. per-flow path search), and optionally the
    GDH-DPQ dual solve on the same instance. Reports per-routing-table time and
    speedup. The dual is slow (~min); enable with run_dual=True."""
    import time
    cfg = cfg or policy.cfg
    inst = Instance.from_globals(g, cfg=cfg)
    policy.eval()
    # warm up (JIT/caches)
    with torch.no_grad():
        h, gw, bv, e2i = policy.encode(inst)
        policy.rollout(inst, h, gw, bv, e2i, greedy=True)

    enc, full = [], []
    with torch.no_grad():
        for _ in range(n_repeats):
            t = time.perf_counter(); policy.encode(inst); enc.append(time.perf_counter() - t)
        for _ in range(n_repeats):
            t = time.perf_counter()
            h, gw, bv, e2i = policy.encode(inst)
            policy.rollout(inst, h, gw, bv, e2i, greedy=True)
            full.append(time.perf_counter() - t)
    policy.train()
    enc_ms, full_ms = np.mean(enc) * 1e3, np.mean(full) * 1e3
    print(f"AI GNN forward pass (neural decision): {enc_ms:9.1f} ms  (+/- {np.std(enc)*1e3:.1f})")
    print(f"AI full routing table (enc + rollout): {full_ms:9.1f} ms  (+/- {np.std(full)*1e3:.1f})")
    print(f"  (rollout = per-flow path search; same search the dual runs each iter)")

    dual_ms = None
    if run_dual and "run_gdh_dpq" in g:
        t = time.perf_counter()
        g["run_gdh_dpq"](g["V_fixed"])
        dual_ms = (time.perf_counter() - t) * 1e3
        print(f"GDH-DPQ dual solve (one routing table): {dual_ms:9.1f} ms")
        print(f"  speedup, full AI vs dual:             {dual_ms/full_ms:9.0f}x")
        print(f"  speedup, neural-only vs dual:         {dual_ms/enc_ms:9.0f}x")
    return dict(encode_ms=enc_ms, full_ms=full_ms, dual_ms=dual_ms)


def policy_scale_sweep(policy, g, scales=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                       cfg=None, name="proposed_ai", train_band=None, log=True):
    """Evaluate an ALREADY-TRAINED policy across traffic scales on the real
    instance. Reports PLR, served-ratio, avg delay (ms), hops, feasibility per
    scale, and prints MATLAB-style arrays to overlay on your benchmark plots.
    If train_band=(lo,hi) is given, scales outside it are flagged as HELD-OUT
    load (=> the sweep doubles as a load-generalization test)."""
    cfg = cfg or policy.cfg
    rho_base = np.asarray(g.get("rho_base", g["rho"]), dtype=float).copy()
    rows = []
    for sc in scales:
        inst = Instance.from_globals(g, cfg=cfg)
        inst.rho = rho_base * float(sc)
        r = evaluate(policy, inst)
        held = (train_band is not None and not (train_band[0] <= sc <= train_band[1]))
        rows.append(dict(scale=float(sc), plr=r["plr"], sr=r["served_ratio"],
                         avgD_ms=r["avgD_ms"], hops=r["avg_hops"],
                         feas=r["feas"], held_out=held))
    if log:
        print(f"{'scale':>6} | {'PLR':>7} | {'SR':>6} | {'avgD(ms)':>9} | "
              f"{'hops':>5} | feas | load")
        for r in rows:
            tag = "HELD-OUT" if r["held_out"] else "in-band"
            dms = r["avgD_ms"] if np.isfinite(r["avgD_ms"]) else float("nan")
            print(f"{r['scale']:6.2f} | {r['plr']:7.4f} | {r['sr']:6.3f} | "
                  f"{dms:9.1f} | {r['hops']:5.2f} | {str(r['feas']):>4} | {tag}")

        def arr(key, scale=1.0, fmt="{:.6f}"):
            return " ".join(fmt.format(r[key] * scale) for r in rows)
        sx = " ".join(f"{r['scale']:.2f}" for r in rows)
        print(f"\n% --- overlay arrays (match your benchmark naming) ---")
        print(f"{name}_tr_x_scale = [{sx}];")
        print(f"{name}_tr_plr  = [{arr('plr')}];")
        print(f"{name}_tr_sr   = [{arr('sr')}];")
        print(f"{name}_tr_avgD = [{arr('avgD_ms', 1e-3)}];   % seconds")
        print(f"{name}_tr_hops = [{arr('hops')}];")
    return rows


def sample_design(seed, cfg, planes=(5, 10), slots=(5, 10), ngw=(6, 16),
                  cg=(3.0, 8.0), cisl=(2.0, 4.0), vis=(1200.0, 2000.0),
                  scale=(0.8, 1.4)):
    """Sample a random synthetic constellation DESIGN (size, gateways, caps,
    visibility, load). Used to train across a distribution of topologies."""
    r = np.random.default_rng(seed)
    return Instance.synthetic(
        cfg=cfg, seed=seed, traffic_seed=seed + 10007,
        n_planes=int(r.integers(planes[0], planes[1] + 1)),
        n_slots=int(r.integers(slots[0], slots[1] + 1)),
        n_gw=int(r.integers(ngw[0], ngw[1] + 1)),
        Cg_val=float(r.uniform(*cg)), Cisl_val=float(r.uniform(*cisl)),
        vis_km=float(r.uniform(*vis)), scale=float(r.uniform(*scale)))


def run_design_generalization(cfg, test_designs, train_sampler=None,
                              n_val=4, log=True):
    """Train across a DISTRIBUTION of constellation designs (train_sampler(seed)
    -> Instance) and evaluate on held-out fixed test_designs (typically LARGER
    than anything trained on -> size extrapolation). Keep-best uses fresh
    in-distribution val designs (never the test set)."""
    if train_sampler is None:
        train_sampler = lambda seed: sample_design(seed, cfg)

    def train_inst(seed):
        return train_sampler(int(seed))

    val_insts = [train_sampler(90000 + i) for i in range(n_val)]

    policy = ConstructionPolicy(cfg)
    if log:
        print(f"=== CROSS-DESIGN generalization: train on sampled designs, "
              f"test on {len(test_designs)} held-out ===")
    train(policy, cfg, instance_fn=train_inst,
          log_every=max(1, cfg.steps // 12),
          val_insts=val_insts, val_every=max(1, cfg.steps // 20))

    if log:
        print(f"\n  {'design (Ns,Ng)':>16} | {'policy SR':>9} | {'greedy SR':>9} | feas")
    res = []
    for inst in test_designs:
        rp = evaluate(policy, inst)
        rg = greedy_nearest_baseline(inst)
        res.append((inst.Ns, inst.Ng, rp["served_ratio"], rg["served_ratio"], rp["feas"]))
        if log:
            print(f"  {f'({inst.Ns},{inst.Ng})':>16} | {rp['served_ratio']:9.3f} | "
                  f"{rg['served_ratio']:9.3f} | {rp['feas']}")
    if log:
        pol_m = np.mean([r[2] for r in res]); grd_m = np.mean([r[3] for r in res])
        print(f"\n  held-out design mean SR: policy {pol_m:.3f}  greedy {grd_m:.3f}")
        print("  feas=True on all unseen sizes => ARCHITECTURE + feasibility transfer.")
        print("  policy-vs-greedy margin here shows whether QUALITY transfers too;")
        print("  a tie on capacity-bound test designs is inconclusive (no gap to exploit).")
    return dict(results=res, policy=policy)


if False:   # was: if __name__ == "__main__":  (disabled so pasting into a Jupyter cell won't auto-run the synthetic demo)
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = CFG(steps=500, batch=8, hidden=64, K_GW=2, device="cpu")
    policy = ConstructionPolicy(cfg)

    # DECISIVE TEST: coordination-gap instance where greedy-nearest is
    # PROVABLY suboptimal. If the policy can't beat greedy here, the
    # architecture/signal is too weak before touching real data.
    def build(traffic_seed, scale):
        return Instance.coordination_gap(cfg=cfg, seed=0,
                                         traffic_seed=traffic_seed, scale=scale)

    def train_inst(seed):
        r = np.random.default_rng(seed)
        return build(int(r.integers(0, 1 << 30)), float(r.uniform(0.9, 1.2)))

    # analytic reference: what does greedy leave on the table?
    g0 = greedy_nearest_baseline(build(1234, 1.0))
    print("=== coordination-gap test (greedy is provably suboptimal) ===")
    print(f"greedy-nearest @ scale 1.0: SR={g0['served_ratio']:.3f} "
          f"(optimal here is ~1.0)\n")

    hist = train(policy, cfg, instance_fn=train_inst, log_every=50)

    print("\n=== paired eval: policy vs greedy-nearest (24 seeds) ===")
    print(f"{'scale':>5} | {'policy SR':>16} | {'greedy SR':>16} | "
          f"{'Delta (pol-grd)':>18} | {'t':>6}")
    seeds = list(range(2000, 2024))
    for sc in (0.8, 1.0, 1.2, 1.4):
        r = paired_eval(policy, build, sc, seeds)
        sig = "*" if abs(r["tstat"]) >= 2.0 else " "
        print(f"{sc:5.1f} | {r['policy_mean']:.3f} +/- {r['policy_std']:.3f}   | "
              f"{r['greedy_mean']:.3f} +/- {r['greedy_std']:.3f}   | "
              f"{r['diff_mean']:+.4f} +/- {r['diff_std']:.3f} | "
              f"{r['tstat']:+5.1f}{sig}")
    print("\n(* = |t| >= 2. Large positive Delta => policy learned "
          "non-myopic gateway choice.)")

