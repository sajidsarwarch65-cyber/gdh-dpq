"""Temporal routing environment: fixed topology, time-varying demand, bursts that
occupy their path for a duration proportional to volume. The future-load ledger is
the carried state that the single-snapshot problem lacked -- admitting a burst now
reserves capacity across the next `dur` epochs, so early choices constrain later ones."""
import numpy as np, heapq, math

class TemporalRoutingEnv:
    def __init__(self, inst, T=12, rate=8.0, seed=0,
                 base_bursts_per_epoch=40, hotspot_bursts_per_epoch=3,
                 base_vol=(4.,16.), hotspot_vol=(60.,120.), hotspot_sats=None):
        self.inst=inst; self.T=T; self.rate=float(rate)
        self.E=inst.n_edges; self.Ng=inst.Ng; self.Ns=inst.Ns
        self.Cisl=float(inst.Cisl); self.Cg=np.asarray(inst.Cg,float)
        self.rng=np.random.default_rng(seed)
        self.bpe=base_bursts_per_epoch; self.hpe=hotspot_bursts_per_epoch
        self.base_vol=base_vol; self.hot_vol=hotspot_vol
        self.hotspot_sats=(np.asarray(hotspot_sats,int) if hotspot_sats is not None
                           else np.array([],int))
        self._prebuild_arrivals()

    def _prebuild_arrivals(self):
        self.arrivals=[]
        nonhot=np.setdiff1d(np.arange(self.Ns), self.hotspot_sats)
        for t in range(self.T):
            evs=[]
            for _ in range(self.bpe):
                s=int(self.rng.choice(nonhot)); v=float(self.rng.uniform(*self.base_vol))
                evs.append((s,v,False))
            if len(self.hotspot_sats):
                for _ in range(self.hpe):
                    s=int(self.rng.choice(self.hotspot_sats)); v=float(self.rng.uniform(*self.hot_vol))
                    evs.append((s,v,True))
            self.arrivals.append(evs)

    def reset(self):
        self.t=0
        self.L_isl=np.zeros((self.E,self.T+64))
        self.L_gw =np.zeros((self.Ng,self.T+64))
        self.delivered=0.0; self.offered=0.0; self.dropped=0.0
        self.delay_wsum=0.0
        return self._obs()

    def _obs(self):
        return dict(t=self.t,
                    isl_free=self.Cisl-self.L_isl[:,self.t],
                    gw_free=self.Cg-self.L_gw[:,self.t],
                    arrivals=self.arrivals[self.t] if self.t<self.T else [])

    def edge_ids(self, path):
        em=self.inst.eidx_mat
        return [int(em[a,b]) for a,b in zip(path[:-1],path[1:])]

    def reserve(self, eids, g, vol):
        dur=max(1,int(math.ceil(vol/self.rate)))
        t0=self.t; t1=self.t+dur
        for e in eids: self.L_isl[e,t0:t1]+=self.rate
        self.L_gw[g,t0:t1]+=self.rate
        return dur

    def path_free_over(self, eids, g, dur):
        t0=self.t; t1=self.t+dur
        m=float((self.Cg[g]-self.L_gw[g,t0:t1]).min())
        for e in eids:
            m=min(m, float((self.Cisl-self.L_isl[e,t0:t1]).min()))
        return m

    def commit(self, src, eids, g, vol):
        dur=max(1,int(math.ceil(vol/self.rate)))
        fit=0
        for k in range(dur):
            tk=self.t+k
            free=self.Cg[g]-self.L_gw[g,tk]
            for e in eids: free=min(free, self.Cisl-self.L_isl[e,tk])
            if free>=self.rate-1e-9: fit+=1
            else: break
        if fit<=0: return 0.0
        t0=self.t; t1=self.t+fit
        for e in eids: self.L_isl[e,t0:t1]+=self.rate
        self.L_gw[g,t0:t1]+=self.rate
        take=min(vol, self.rate*fit)
        self.delivered+=take
        return take

    def advance(self):
        self.t+=1
        return self.t>=self.T
