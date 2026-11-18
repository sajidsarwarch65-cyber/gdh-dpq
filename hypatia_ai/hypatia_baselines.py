import numpy as np
from collections import defaultdict

D_TX_ISL=0.0002; D_PROC_ISL=0.0001; D_TX_SG=0.0005; D_PROC_SG=0.0002
D_MAX=0.5; L_MIN=30.0; KAPPA=0.5
CADSP_GW_PEN=1.0; CADSP_ISL_PEN=0.5
DBPR_ALPHA_Q=1.0; DBPR_BETA_DIST=0.25; DBPR_ITERS=5
c_light=299792.458

_I=None; Ns=Ng=None; rho=V=NBR_LIST=sat_positions=None
Cisl=None; Cg=None; isl_lifetime_s=None

def _bind(inst):
    global _I,Ns,Ng,rho,V,NBR_LIST,sat_positions,Cisl,Cg,isl_lifetime_s
    _I=inst; Ns=inst.Ns; Ng=inst.Ng
    rho=np.asarray(inst.rho,float); V=inst.V; NBR_LIST=inst.NBR_LIST
    sat_positions=inst.sat_positions; Cisl=float(inst.Cisl)
    Cg=np.asarray(inst.Cg,float); isl_lifetime_s=inst.isl_lifetime_s

def isl_prop_s(i,j): return _I.isl_prop_s(i,j)
def sg_prop_s(s,g): return _I.sg_prop_s(s,g)
def edge_key(i,j): return _I.edge_key(i,j)
def sat_gw_km(s,g): return _I.sat_gw_km(s,g)
def dijkstra(s,t,wf): return _I.dijkstra(s,t,wf)
def exit_sats(V_,g): return np.where(V_[:,g]>0)[0]
def any_exit(V_): return np.where(V_.sum(axis=1)>0)[0]

def is_path_valid(path,g):
    if path is None or g is None or g<0: return False
    d=0.0
    for a,b in zip(path[:-1],path[1:]):
        d+=isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL
        if float(isl_lifetime_s[a,b])<L_MIN: return False
    d+=sg_prop_s(path[-1],g)+D_TX_SG+D_PROC_SG
    if d>D_MAX: return False
    return True

def full_delay(path,g,gwl,isll): return _I.full_delay(path,g,gwl,isll)

def cost_dsp(path,g):
    c=0.0
    for a,b in zip(path[:-1],path[1:]): c+=isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL
    return c+sg_prop_s(path[-1],g)+D_TX_SG+D_PROC_SG

def cost_cadsp(path,g,gwl,isll):
    c=0.0
    for a,b in zip(path[:-1],path[1:]):
        c+=isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL+CADSP_ISL_PEN*(isll[edge_key(a,b)]/max(Cisl,1e-12))
    return c+sg_prop_s(path[-1],g)+D_TX_SG+D_PROC_SG+CADSP_GW_PEN*(gwl[g]/max(Cg[g],1e-12))

def cost_dbpr_for_kappa(path,g):
    c=0.0
    for a,b in zip(path[:-1],path[1:]): c+=isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL
    return c+sg_prop_s(path[-1],g)+D_TX_SG+D_PROC_SG

def cost_gpsr(path,g):
    c=0.0
    for a,b in zip(path[:-1],path[1:]): c+=isl_prop_s(a,b)+D_TX_ISL+D_PROC_ISL
    return c+sg_prop_s(path[-1],g)+D_TX_SG+D_PROC_SG

def cost_directdrop(s,g): return sg_prop_s(s,g)+D_TX_SG+D_PROC_SG

def admit_partial(path,g,flow,gwl,isll):
    head_gw=Cg[g]-gwl[g]
    if head_gw<=1e-12: return 0.0
    take=min(flow,head_gw)
    for a,b in zip(path[:-1],path[1:]):
        head_e=Cisl-isll[edge_key(a,b)]
        if head_e<=1e-12: return 0.0
        take=min(take,head_e)
    if take<=1e-12: return 0.0
    gwl[g]+=take
    for a,b in zip(path[:-1],path[1:]): isll[edge_key(a,b)]+=take
    return take

def _commit_choice(s,flow,cand,gwl,isll,paths,cgw,hc,sf):
    use=None
    if cand:
        cand=sorted(cand,key=lambda t:t[0]); use=(cand[0][1],cand[0][2])
    if use is None: return 0.0
    path,g=use; take=admit_partial(path,g,flow,gwl,isll)
    if take<=1e-12: return 0.0
    paths[s]=[int(x) for x in path]; cgw[s]=int(g); hc[s]=max(len(path)-1,0); sf[s]=take
    return take

def pack_kpi(scheme,paths,cgw,hc,gwl,isll,fd,served_flow=None):
    rho_l=np.asarray(rho,float); ridx=np.array(sorted(paths.keys()),dtype=int)
    offered=float(np.sum(rho_l))
    if served_flow is not None:
        served=float(sum(served_flow.get(int(s),0.0) for s in ridx))
    else:
        served=float(np.sum(rho_l[ridx])) if len(ridx) else 0.0
    sr=served/offered if offered>0 else 0.0
    ds=[fd[s] for s in ridx if not np.isnan(fd[s])]
    hs=[hc[s] for s in ridx if hc[s]>0]
    return dict(scheme=scheme,served_ratio=sr,plr=1-sr,
                avgD_ms=float(np.mean(ds))*1000 if ds else 0.0,
                avg_hops=float(np.mean(hs)) if hs else 0.0,
                paths={int(s):[int(x) for x in paths[s]] for s in paths},
                served_flow=({int(s):float(served_flow[s]) for s in served_flow} if served_flow is not None else None))

def run_direct_drop(inst):
    _bind(inst)
    cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int); gwl=np.zeros(Ng); isll=defaultdict(float); paths={}; sf={}
    for s in range(Ns):
        opts=[]
        for g in np.where(V[s]>0)[0]:
            g=int(g)
            if not is_path_valid([s],g): continue
            opts.append((cost_directdrop(s,g),[s],g))
        _commit_choice(s,float(rho[s]),opts,gwl,isll,paths,cgw,hc,sf)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("Direct Drop",paths,cgw,hc,gwl,isll,fd,served_flow=sf)

def run_dsp(inst):
    _bind(inst)
    cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int); gwl=np.zeros(Ng); isll=defaultdict(float); paths={}; sf={}
    for s in range(Ns):
        vis=np.where(V[s]>0)[0]; opts=[]
        if len(vis):
            for g in vis:
                g=int(g)
                if not is_path_valid([s],g) or cost_dsp([s],g)>KAPPA: continue
                opts.append((cost_dsp([s],g),[s],g))
        else:
            for g in np.where(V.sum(axis=0)>0)[0]:
                g=int(g); ex=exit_sats(V,g)
                if not len(ex): continue
                wf=lambda u,v: isl_prop_s(u,v)+D_TX_ISL+D_PROC_ISL
                path=dijkstra(s,ex,wf)
                if path is None or not is_path_valid(path,g) or cost_dsp(path,g)>KAPPA: continue
                opts.append((cost_dsp(path,g),[int(x) for x in path],g))
        _commit_choice(s,float(rho[s]),opts,gwl,isll,paths,cgw,hc,sf)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("DSP",paths,cgw,hc,gwl,isll,fd,served_flow=sf)

def run_cadsp(inst):
    _bind(inst)
    cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int); gwl=np.zeros(Ng); isll=defaultdict(float); paths={}; sf={}
    for s in range(Ns):
        vis=np.where(V[s]>0)[0]; opts=[]
        if len(vis):
            for g in vis:
                g=int(g)
                if not is_path_valid([s],g) or cost_cadsp([s],g,gwl,isll)>KAPPA: continue
                opts.append((cost_cadsp([s],g,gwl,isll),[s],g))
        else:
            for g in np.where(V.sum(axis=0)>0)[0]:
                g=int(g); ex=exit_sats(V,g)
                if not len(ex): continue
                wf=lambda u,v: isl_prop_s(u,v)+D_TX_ISL+D_PROC_ISL+CADSP_ISL_PEN*(isll[edge_key(u,v)]/max(Cisl,1e-12))
                path=dijkstra(s,ex,wf)
                if path is None or not is_path_valid(path,g) or cost_cadsp(path,g,gwl,isll)>KAPPA: continue
                opts.append((cost_cadsp(path,g,gwl,isll),[int(x) for x in path],g))
        _commit_choice(s,float(rho[s]),opts,gwl,isll,paths,cgw,hc,sf)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("CA-DSP",paths,cgw,hc,gwl,isll,fd,served_flow=sf)

def run_dbpr(inst):
    _bind(inst)
    cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int); gwl=np.zeros(Ng); isll=defaultdict(float); paths={}; sf={}
    if not len(any_exit(V)):
        fd=np.full(Ns,np.nan); return pack_kpi("DBPR",{},cgw,hc,gwl,isll,fd)
    sq=np.zeros(Ns)
    for _ in range(DBPR_ITERS):
        paths={}; cgw[:]=-1; hc[:]=0; gwl[:]=0.; isll=defaultdict(float)
        for s in range(Ns):
            vis=np.where(V[s]>0)[0]
            if len(vis):
                vok=[int(g) for g in vis if is_path_valid([s],int(g))]
                if not vok: continue
                bg=int(min(vok,key=lambda g:DBPR_ALPHA_Q*gwl[g]+DBPR_BETA_DIST*sat_gw_km(s,g)))
                paths[s]=[s]; cgw[s]=bg; gwl[bg]+=rho[s]; continue
            bc=np.inf; bp=None; bg=None
            for g in np.where(V.sum(axis=0)>0)[0]:
                g=int(g); ex=exit_sats(V,g)
                if not len(ex): continue
                wf=lambda u,v: DBPR_ALPHA_Q*(sq[v]+isll[edge_key(u,v)])+DBPR_BETA_DIST*(isl_prop_s(u,v)*c_light)
                path=dijkstra(s,ex,wf)
                if path is None or not is_path_valid(path,g): continue
                cost=(sum(DBPR_ALPHA_Q*(sq[b]+isll[edge_key(a,b)])+DBPR_BETA_DIST*(isl_prop_s(a,b)*c_light) for a,b in zip(path[:-1],path[1:]))+DBPR_ALPHA_Q*gwl[g]+DBPR_BETA_DIST*sat_gw_km(path[-1],g))
                if cost<bc: bc=cost; bp=path; bg=g
            if bp is None: continue
            paths[s]=[int(x) for x in bp]; cgw[s]=bg; hc[s]=max(len(bp)-1,0); gwl[bg]+=rho[s]
            for a,b in zip(bp[:-1],bp[1:]): isll[edge_key(a,b)]+=rho[s]
        sq[:]=0.
        for s,p in paths.items():
            for n in p[1:]: sq[n]+=rho[s]
    fp={}; fg=np.zeros(Ng); fi=defaultdict(float); s2={}; cgw2=-np.ones(Ns,int); hc2=np.zeros(Ns,int)
    for s in sorted(paths.keys()):
        p=paths[s]; g=int(cgw[s])
        if not is_path_valid(p,g) or cost_dbpr_for_kappa(p,g)>KAPPA: continue
        take=admit_partial(p,g,float(rho[s]),fg,fi)
        if take<=1e-12: continue
        fp[s]=p; cgw2[s]=g; hc2[s]=hc[s]; s2[s]=take
    fd=np.full(Ns,np.nan)
    for s,p in fp.items(): fd[s]=full_delay(p,cgw2[s],fg,fi)
    return pack_kpi("DBPR",fp,cgw2,hc2,fg,fi,fd,served_flow=s2)

def run_gpsr(inst):
    _bind(inst)
    paths={}; cgw=-np.ones(Ns,int); hc=np.zeros(Ns,int); gwl=np.zeros(Ng); isll=defaultdict(float); sf={}
    eany=any_exit(V); MAX_H=max(10,Ns)
    def gpsr_next(u,tgt,vis):
        nbrs=NBR_LIST[u]
        if not len(nbrs): return None
        bv=None; bd=np.linalg.norm(sat_positions[u]-tgt)
        for v in nbrs:
            v=int(v)
            if v in vis: continue
            dv=np.linalg.norm(sat_positions[v]-tgt)
            if dv<bd-1e-12: bd=dv; bv=v
        return bv
    for s in range(Ns):
        flow=float(rho[s])
        if V[s].sum()>0:
            vg=sorted(np.where(V[s]>0)[0],key=lambda g:sat_gw_km(s,g)); opts=[]
            for g in vg:
                g=int(g)
                if not is_path_valid([s],g) or cost_gpsr([s],g)>KAPPA: continue
                opts.append((cost_gpsr([s],g),[s],g))
            _commit_choice(s,flow,opts,gwl,isll,paths,cgw,hc,sf); continue
        if not len(eany): continue
        d2=np.sum((sat_positions[eany]-sat_positions[s])**2,axis=1)
        tgt=sat_positions[int(eany[np.argmin(d2)])]
        path=[s]; vis={s}; cur=s; ok=False
        for _ in range(MAX_H):
            if V[cur].sum()>0: ok=True; break
            nx=gpsr_next(cur,tgt,vis)
            if nx is None: break
            path.append(nx); vis.add(nx); cur=nx
        if not ok and V[cur].sum()==0: continue
        vg=sorted(np.where(V[cur]>0)[0],key=lambda g:sat_gw_km(cur,g)); opts=[]
        for g in vg:
            g=int(g)
            if not is_path_valid(path,g) or cost_gpsr(path,g)>KAPPA: continue
            opts.append((cost_gpsr(path,g),[int(x) for x in path],g))
        _commit_choice(s,flow,opts,gwl,isll,paths,cgw,hc,sf)
    fd=np.full(Ns,np.nan)
    for s,p in paths.items(): fd[s]=full_delay(p,cgw[s],gwl,isll)
    return pack_kpi("GPSR",paths,cgw,hc,gwl,isll,fd,served_flow=sf)

ALL_BASELINES={"Direct Drop":run_direct_drop,"DSP":run_dsp,"CA-DSP":run_cadsp,"DBPR":run_dbpr,"GPSR":run_gpsr}
