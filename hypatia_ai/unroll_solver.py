# Unrolled K-step policy: wraps neural_solver_MCpath, adds 8-dim node features
# (isl_load_frac + unserved_frac), residual routing per step, REINFORCE across steps.
# Keeps neural_solver_MCpath.py UNTOUCHED.
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, dataclasses
sys.path.insert(0,'.')
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP

# ---- 8-dim graph tensors: reuse MP builder, append 2 load-state columns ----
def build_gt8(inst, isl_load, unserved, dev):
    gt = MP.build_graph_tensors(inst, dev)
    Ns = inst.Ns
    satload = np.zeros(Ns, dtype=np.float32)
    for k,(i,j) in enumerate(inst.isl_edges):
        satload[i]+=isl_load[k]; satload[j]+=isl_load[k]
    denom = np.maximum(inst.deg,1.0)*float(inst.Cisl)
    satload = np.clip(satload/denom, 0, 1)
    unfrac = np.clip(unserved/np.maximum(inst.rho,1e-9), 0, 1).astype(np.float32)
    extra = torch.tensor(np.stack([satload, unfrac],axis=1).astype(np.float32), device=dev)  # (Ns,2)
    gt["X"] = torch.cat([gt["X"].float(), extra.float()], dim=1)   # (Ns,8)
    # ---- predictive lookahead node features (t+120s) ----
    Vn = getattr(inst, "V_next", inst.V)
    rn = getattr(inst, "rho_next", inst.rho)
    has_vis_next = (np.asarray(Vn).sum(axis=1) > 0).astype(np.float32)
    rho_next_n = np.clip(np.asarray(rn,dtype=np.float32)/max(float(inst.rho.max()),1e-9), 0, 1).astype(np.float32)
    extra2 = torch.tensor(np.stack([has_vis_next, rho_next_n],axis=1).astype(np.float32), device=dev)  # (Ns,2)
    gt["X"] = torch.cat([gt["X"].float(), extra2.float()], dim=1)   # (Ns,10)
    return gt

# ---- widen encoder node_in 6->8 (copy existing 6 weights, zero-init the 2 new) ----
def widen_encoder(pol, dev):
    enc = pol.enc
    old = enc.node_in                       # Linear(6,H)
    H = old.out_features
    in_now = old.in_features                    # 6 (cold) or 8 (already widened once)
    new = nn.Linear(10, H).to(dev)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :in_now] = old.weight   # keep learned cols, new start at 0
        new.bias.copy_(old.bias)
    enc.node_in = new
    return pol

# ---- one encode with an 8-dim gt ----
def encode8(pol, gt, cfg, dev):
    h = F.relu(pol.enc.node_in(gt["X"]))
    for layer in pol.enc.layers: h = layer(h, gt["edge_index"], gt["edge_attr"])
    gw_pool = gt["Wvis"] @ h
    gw_emb = pol.enc.gw_proj(torch.cat([gw_pool, gt["gw_static"]], dim=-1))
    if cfg.USE_LEARNED_EDGE_BIAS and gt["und_u"].numel()>0:
        eu,ev = gt["und_u"], gt["und_v"]
        bias_vec = F.softplus(pol.enc.bias_head(torch.cat([h[eu],h[ev],gt["und_attr"]],dim=-1))).squeeze(-1)*1e-3
    else:
        bias_vec = torch.zeros(gt["und_u"].numel(), device=dev)
    return h, gw_emb, bias_vec, gt["edge_to_idx"]

# ---- K-step UNROLLED rollout: route residual demand each step ----
def unrolled(pol, inst, K, greedy, broken_links=None):
    cfg=inst.cfg; dev=getattr(cfg,'device','cpu')
    isl_load=np.zeros(inst.n_edges); gwl=np.zeros(inst.Ng)
    unserved=inst.rho.copy().astype(np.float32)
    full_sol={}; all_logps=[]
    srs=[]
    rho0_sum=inst.rho.sum()
    orig_rho=inst.rho.copy()
    for k in range(K):
        # route only residual demand: temporarily set inst.rho = unserved
        inst.rho = unserved.copy()
        gt = build_gt8(inst, isl_load, unserved, dev)
        h,gw_emb,bias_vec,e2i = encode8(pol, gt, cfg, dev)
        logp,ent,sol,gwl2,isll = pol.rollout(inst, h, gw_emb, bias_vec, e2i, greedy=greedy, gwl_init=gwl, isl_load_init=isl_load, broken_links=broken_links)
        if logp is not None and torch.is_tensor(logp) and logp.requires_grad:
            all_logps.append(logp)
        # update served: subtract what this step delivered
        for s,gd in (sol.items() if hasattr(sol,'items') else []):
            if isinstance(gd,dict):
                d=sum(v[0] for v in gd.values())
                unserved[s]=max(unserved[s]-d, 0.0)
                full_sol.setdefault(s,{})
                for g,v in gd.items(): full_sol[s][g]=v
        # rebuild isl_load from isll dict
        isl_load=np.zeros(inst.n_edges)
        for ek,v in (isll.items() if hasattr(isll,'items') else []):
            idx=e2i.get(ek)
            if idx is not None: isl_load[idx]=v
        gwl=gwl2
        served_sofar = rho0_sum - unserved.sum()
        srs.append(served_sofar / rho0_sum)
    inst.rho = orig_rho
    return srs, full_sol, all_logps

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",default="gate"); ap.add_argument("--K",type=int,default=4)
    ap.add_argument("--steps",type=int,default=120); a=ap.parse_args()

    t0=time.time()
    inst,_,_=HP.build_instance(scale=1.0, hotspot=True, use_std_gw=True)
    cfg=inst.cfg
    try: cfg=dataclasses.replace(cfg, USE_LEARNED_EDGE_BIAS=True)
    except Exception: cfg.USE_LEARNED_EDGE_BIAS=True
    for k,v in [('PATH_TOPK',4),('PATH_NDET',2)]:
        try: setattr(cfg,k,v)
        except: pass
    inst.cfg=cfg; dev=getattr(cfg,'device','cpu')
    print("[init] built %.0fs offered=%.0f K=%d" % (time.time()-t0, inst.rho.sum(), a.K), flush=True)

    pol=MP.ConstructionPolicy(cfg).to(dev)
    pol=widen_encoder(pol, dev)

    # SNR GATE: does SR climb across steps (untrained, greedy decode)?
    srs,_,_=unrolled(pol, inst, K=a.K, greedy=True)
    print("=== UNROLL SNR GATE (untrained, cumulative SR across steps) ===", flush=True)
    for k,sr in enumerate(srs): print("  after step %d: cumulative SR=%.4f" % (k+1, sr), flush=True)
    print("  greedy=0.763 dual=0.918  |  step1→K delta=%+.4f" % (srs[-1]-srs[0]), flush=True)

    if a.mode=="train":
        opt=torch.optim.Adam(pol.parameters(), lr=1e-3)
        base=srs[-1]
        print("[train] start, greedy-unroll baseline SR=%.4f" % base, flush=True)
        for step in range(a.steps):
            # sampled trajectory
            srs_s, sol_s, logps = unrolled(pol, inst, K=a.K, greedy=False)
            R = srs_s[-1]
            # self-critical baseline: greedy unroll
            with torch.no_grad():
                srs_g,_,_ = unrolled(pol, inst, K=a.K, greedy=True)
            adv = R - srs_g[-1]
            if logps:
                loss = -(adv) * torch.cat([lp.view(-1) for lp in logps]).sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            if step%10==0:
                vg,_,_=unrolled(pol, inst, K=a.K, greedy=True)
                print("[%3d] R_samp=%.4f adv=%+.4f val_SR=%.4f" % (step, R, adv, vg[-1]), flush=True)
        torch.save(pol.state_dict(),"policy_unroll.pt")
        vg,_,_=unrolled(pol, inst, K=a.K, greedy=True)
        print("[done] final unroll val_SR=%.4f vs greedy 0.763 dual 0.918" % vg[-1], flush=True)
