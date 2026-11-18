"""Port of v6e single-stage joint dual to a Hypatia-Instance interface."""
import numpy as np, heapq, math
from collections import defaultdict

alpha1=1.0; alpha2=0.02
_LOG_CONV=False
_BROKEN=set()
_CONV_LOG=[]
D_TX_ISL=0.0002; D_PROC_ISL=0.0001; D_TX_SG=0.0005; D_PROC_SG=0.0002
c_light=299792.458
QUEUE_SCALE=0.003; QUEUE_MAX=0.03; QUEUE_EPS=1e-3
ALPHA4=0.0001; L_MAX_LIFE=600.0; L_MIN=0.0; D_MAX=1e9
ALPHA_HO=0.02; THO_S=60.0; KAPPA=0.5; FB_HOP_PEN=0.0001; TIE_EPS=1e-12
MAX_SPLIT_PATHS=8
DUAL_MAX_ITERS=80; DUAL_EARLY_STOP=20
LAM_MAX=100.0; NU_MAX=100.0; eta_load=0.25
DUAL_STEP0=0.15; DUAL_STEP_DECAY=0.03

def queue_delay(load,cap):
    cap=float(cap); load=max(0.,float(load))
    rho=min(load/max(cap,1e-12),1.-QUEUE_EPS)
    wq=QUEUE_SCALE*rho/max(cap*(1.-rho),1e-9)
    return min(wq,QUEUE_MAX)

def run_gdh_v6e(inst, DST, GW_SERVES, use_lambda=True, use_nu=True, trace=False, lam_init=None, nu_init=None, return_paths=False):
    Ns,Ng=inst.Ns,inst.Ng
    V=inst.V; Cg=inst.Cg; Cisl=float(inst.Cisl); rho=inst.rho
    NBR=inst.NBR_LIST; life=inst.isl_lifetime_s
    ekey=inst.edge_key
    isl_prop=inst.isl_prop_s; sg_prop=inst.sg_prop_s
    SUPER=Ns+Ng

    def is_valid(path,g):
        if path is None or g is None or g<0: return False
        for a,b in zip(path[:-1],path[1:]):
            if float(life[a,b])<L_MIN: return False
        return True
    def cost_dsp(path,g):
        c=0.
        for a,b in zip(path[:-1],path[1:]): c+=isl_prop(a,b)+D_TX_ISL+D_PROC_ISL
        c+=sg_prop(path[-1],g)+D_TX_SG+D_PROC_SG
        return c
    def full_delay(path,g,gwl,isll):
        tot=0.
        for a,b in zip(path[:-1],path[1:]):
            e=ekey(a,b); tot+=isl_prop(a,b)+D_TX_ISL+D_PROC_ISL+queue_delay(isll.get(e,0.),Cisl)
        tot+=sg_prop(path[-1],g)+D_TX_SG+D_PROC_SG+queue_delay(gwl[g],Cg[g])
        return tot
    def path_headroom(path,g,gwl,isll):
        head=max(Cg[g]-gwl[g],0.)
        for a,b in zip(path[:-1],path[1:]):
            head=min(head,max(Cisl-isll[ekey(a,b)],0.))
        return head
    def admit_partial(path,g,flow,gwl,isll):
        take=min(flow,path_headroom(path,g,gwl,isll))
        if take<=1e-9: return 0.
        gwl[g]+=take
        for a,b in zip(path[:-1],path[1:]): isll[ekey(a,b)]+=take
        return take

    def _path_for(s,lam,nu,fgwl,fisll,with_nu):
        dv=np.full(SUPER+1,np.inf); par=np.full(SUPER+1,-1,int)
        dv[s]=0.; pq=[(0.,s)]
        while pq:
            du,u=heapq.heappop(pq)
            if du!=dv[u]: continue
            if u==SUPER: break
            if Ns<=u<Ns+Ng:
                if du<dv[SUPER]: dv[SUPER]=du; par[SUPER]=u; heapq.heappush(pq,(du,SUPER))
                continue
            for v in NBR[u]:
                v=int(v); ek=ekey(u,v)
                if ek in _BROKEN: continue
                if fisll[ek]>=Cisl-1e-9: continue
                stab=ALPHA4*max(0.,L_MAX_LIFE-float(life[u,v]))
                w=isl_prop(u,v)+D_TX_ISL+D_PROC_ISL+FB_HOP_PEN+stab+(float(nu.get(ek,0.)) if with_nu else 0.)
                w+=TIE_EPS*(u*Ns+v); nd=du+w
                if nd<dv[v]: dv[v]=nd; par[v]=u; heapq.heappush(pq,(nd,v))
            for g in np.where(V[u]>0)[0]:
                g=int(g); gn=Ns+g
                if fgwl[g]>=Cg[g]-1e-9: continue
                if not GW_SERVES[g,DST[s]]: continue
                w=sg_prop(u,g)+D_TX_SG+D_PROC_SG+float(lam[g])
                w+=TIE_EPS*(Ns*Ns+u*Ng+g); nd=du+w
                if nd<dv[gn]: dv[gn]=nd; par[gn]=u; heapq.heappush(pq,(nd,gn))
        if not np.isfinite(dv[SUPER]): return None,None
        node=SUPER; rev=[]
        while node!=-1: rev.append(node); node=par[node]
        rev.reverse(); rev=rev[:-1]; gn=int(rev[-1])
        if gn<Ns: return None,None
        sp=[int(x) for x in rev if int(x)<Ns]
        if not sp: return None,None
        return sp,gn-Ns

    def _route(lam,nu,prev_gw,prev_isl,use_q,with_nu):
        qg=np.zeros(Ng); q_isl={}
        if use_q:
            for g in range(Ng): qg[g]=queue_delay(prev_gw[g],Cg[g])
            for i in range(Ns):
                for j in NBR[i]:
                    if j<=i: continue
                    q_isl[ekey(i,j)]=queue_delay(float(prev_isl.get(ekey(i,j),0.)),Cisl)
        gwl_=np.zeros(Ng); isll_=defaultdict(float)
        for s in range(Ns):
            rs=float(rho[s])
            dv=np.full(SUPER+1,np.inf); par=np.full(SUPER+1,-1,int)
            dv[s]=0.; pq=[(0.,s)]
            while pq:
                du,u=heapq.heappop(pq)
                if du!=dv[u]: continue
                if u==SUPER: break
                if Ns<=u<Ns+Ng:
                    if du<dv[SUPER]: dv[SUPER]=du; par[SUPER]=u; heapq.heappush(pq,(du,SUPER))
                    continue
                for v in NBR[u]:
                    v=int(v); ek=ekey(u,v)
                    L=float(life[u,v]); pen=ALPHA_HO*(1. if L<=THO_S else 0.)
                    stab=ALPHA4*max(0.,L_MAX_LIFE-L)
                    Dij=isl_prop(u,v)+D_TX_ISL+D_PROC_ISL+(float(q_isl.get(ek,0.)) if use_q else 0.)
                    w=rs*(alpha1*Dij+alpha2+pen+stab+(float(nu.get(ek,0.)) if with_nu else 0.))
                    w+=TIE_EPS*(u*Ns+v); nd=du+w
                    if nd<dv[v]: dv[v]=nd; par[v]=u; heapq.heappush(pq,(nd,v))
                for g in np.where(V[u]>0)[0]:
                    g=int(g); gn=Ns+g
                    if not GW_SERVES[g,DST[s]]: continue
                    Dig=sg_prop(u,g)+D_TX_SG+D_PROC_SG+(float(qg[g]) if use_q else 0.)
                    w=rs*(alpha1*Dig+float(lam[g]))
                    w+=TIE_EPS*(Ns*Ns+u*Ng+g); nd=du+w
                    if nd<dv[gn]: dv[gn]=nd; par[gn]=u; heapq.heappush(pq,(nd,gn))
            if not np.isfinite(dv[SUPER]): continue
            node=SUPER; rev=[]
            while node!=-1: rev.append(node); node=par[node]
            rev.reverse(); rev=rev[:-1]; gn=int(rev[-1])
            if gn<Ns: continue
            g_best=gn-Ns; sp=[int(x) for x in rev if int(x)<Ns]
            if not sp or not is_valid(sp,g_best): continue
            if len(sp)>=2:
                for a,b in zip(sp[:-1],sp[1:]): isll_[ekey(a,b)]+=rs
            gwl_[g_best]+=rs
        return gwl_,isll_

    def _admit(lam,nu,with_nu):
        fp={}; fgwl=np.zeros(Ng); fisll=defaultdict(float); served={}
        d_wsum=0.
        for s in sorted(range(Ns),key=lambda s:-rho[s]):
            remaining=float(rho[s]); taken=0.
            for _leg in range(MAX_SPLIT_PATHS):
                if remaining<=1e-9: break
                p,g=_path_for(s,lam,nu,fgwl,fisll,with_nu)
                if p is None or not is_valid(p,g) or cost_dsp(p,g)>KAPPA: break
                take=admit_partial(p,g,remaining,fgwl,fisll)
                if take<=1e-9: break
                remaining-=take; taken+=take
                d_wsum+=take*full_delay(p,g,fgwl,fisll)
                if s not in fp: fp[s]=p
            if taken>1e-9: served[s]=taken
        tot=float(sum(served.values()))
        avg_d=(d_wsum/tot) if tot>1e-9 else float('inf')
        return fp,fgwl,fisll,served,tot,avg_d

    lam=np.zeros(Ng) if lam_init is None else np.asarray(lam_init,float).copy()
    nu=defaultdict(float)
    if nu_init is not None:
        for _k,_v in (nu_init.items() if hasattr(nu_init,'items') else []): nu[_k]=float(_v)
    pg=np.zeros(Ng); pi=defaultdict(float)
    best_lam=lam.copy(); best_nu={}; best_served=-1.; best_avgd=float('inf'); no_gain=0
    do_ascend=bool(use_lambda or use_nu); n_iters=DUAL_MAX_ITERS if do_ascend else 1
    for it in range(n_iters):
        gwl_,isll_=_route(lam,nu,pg,pi,True,use_nu)
        fp,fgwl,fisll,served,served_now,avgd_now=_admit(lam,nu,use_nu)
        if globals().get('_LOG_CONV'):
            _mgw=float(np.maximum(pg-Cg,0.).max()) if len(pg) else 0.
            _misl=float(max((l-Cisl for l in pi.values()), default=0.))
            _tv=float(sum(max(l-Cisl,0.) for l in pi.values())+float(np.maximum(pg-Cg,0.).sum()))
            _nzn=sum(1 for v in nu.values() if v>1e-9)
            _CONV_LOG.append((_mgw,_misl,_nzn,_tv,served_now,avgd_now*1e3))
        TOL=1e-6*max(float(np.sum(rho)),1.)
        if (served_now>best_served+TOL or (served_now>best_served-TOL and avgd_now<best_avgd-1e-12)):
            best_served=max(best_served,served_now); best_avgd=avgd_now
            best_lam=lam.copy(); best_nu=dict(nu); no_gain=0
        else: no_gain+=1
        if not do_ascend: break
        if no_gain>=DUAL_EARLY_STOP: break
        pg=(1.-eta_load)*pg+eta_load*gwl_
        nprev=defaultdict(float)
        for e in set(pi)|set(isll_): nprev[e]=(1.-eta_load)*float(pi.get(e,0.))+eta_load*float(isll_.get(e,0.))
        pi=nprev
        alpha_t=DUAL_STEP0/(1.+DUAL_STEP_DECAY*it)
        if use_lambda: lam=np.clip(lam+alpha_t*(np.maximum(pg-Cg,0.)/Cg),0.,LAM_MAX)
        if use_nu:
            for e,l in pi.items():
                ov=float(l-Cisl)
                if ov>1e-12: nu[e]=float(np.clip(float(nu.get(e,0.))+alpha_t*(ov/Cisl),0.,NU_MAX))
    fp,fgwl,fisll,served,tot,avgd=_admit(best_lam,best_nu,use_nu)
    offered=float(np.sum(rho))
    _hw=0.; _hden=0.
    try:
        for _s,_p in (fp.items() if hasattr(fp,'items') else []):
            if _p is not None and hasattr(_p,'__len__') and len(_p)>=1:
                _dl=float(served.get(_s,0.))
                _hops=max(len(_p)-1,0)
                _hw+=_dl*_hops; _hden+=_dl
    except Exception:
        pass
    _mean_hops=(_hw/_hden) if _hden>1e-9 else 0.0
    return dict(served_ratio=tot/max(offered,1e-12), plr=1.-tot/max(offered,1e-12),
                served=tot, offered=offered, avgD_ms=avgd*1e3,
                nz_nu=sum(1 for v in best_nu.values() if v>1e-9),
                nz_lam=int(np.sum(best_lam>1e-9)), iters=it+1,
                gwl=np.asarray(fgwl,float).copy(),
                isl_max=float(max(fisll.values()) if len(fisll) else 0.0),
                isl_loads=np.array(sorted(fisll.values(),reverse=True)[:20]) if len(fisll) else np.zeros(0),
                mean_hops=float(_mean_hops),
                paths=(dict(fp) if return_paths else None),
                served_flow=(dict(served) if return_paths else None))
