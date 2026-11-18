import sys, numpy as np, torch, time, dataclasses; sys.path.insert(0,'.')
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP

t0=time.time()
inst,_,_=HP.build_instance(scale=1.0, hotspot=True, use_std_gw=True)
cfg=inst.cfg
try: cfg=dataclasses.replace(cfg, USE_LEARNED_EDGE_BIAS=True)
except Exception: cfg.USE_LEARNED_EDGE_BIAS=True
for k,v in [('PATH_TOPK',4),('PATH_NDET',2)]:
    try: setattr(cfg,k,v)
    except: pass
inst.cfg=cfg
dev=getattr(cfg,'device','cpu')
print("[init] built %.0fs offered=%.0f" % (time.time()-t0, inst.rho.sum()), flush=True)

pol=MP.ConstructionPolicy(cfg).to(dev); pol.eval()

# --- UNROLLED rollout: K steps, feeding back isl_load/gwl as extra node features ---
# We inject load state by MUTATING inst.rho-independent load features via a wrapper:
# rebuild graph tensors each step with 2 extra node-feature columns appended.
def build_gt_with_loads(inst, isl_load, gwl, dev):
    gt = MP.build_graph_tensors(inst, dev)
    Ns = inst.Ns
    # per-sat outgoing ISL load fraction (sum of incident edge loads / (deg*Cisl))
    satload = np.zeros(Ns, dtype=np.float32)
    for k,(i,j) in enumerate(inst.isl_edges):
        satload[i]+=isl_load[k]; satload[j]+=isl_load[k]
    denom = np.maximum(inst.deg,1.0)*float(inst.Cisl)
    satload = satload/denom
    load_col = torch.tensor(satload, device=dev).unsqueeze(-1)
    # append as a 7th node feature (encoder expects node_dim=6, so we FOLD it into X[:,3..5]?)
    # Simplest non-invasive probe: scale the rho feature by (1 - satload) to signal congestion
    gt["X"] = gt["X"].clone()
    gt["X"][:,4] = gt["X"][:,4]*(1.0 - torch.clamp(load_col.squeeze(-1),0,1)*0.5)  # dampen has_vis where congested
    return gt

def unrolled_rollout(pol, inst, K=4, greedy=True):
    dev=getattr(inst.cfg,'device','cpu')
    isl_load=np.zeros(inst.n_edges); gwl=np.zeros(inst.Ng)
    srs=[]
    for k in range(K):
        gt=build_gt_with_loads(inst, isl_load, gwl, dev)
        with torch.no_grad():
            h=torch.relu(pol.enc.node_in(gt["X"]))
            for layer in pol.enc.layers: h=layer(h,gt["edge_index"],gt["edge_attr"])
            gw_pool=gt["Wvis"]@h
            gw_emb=pol.enc.gw_proj(torch.cat([gw_pool,gt["gw_static"]],dim=-1))
            if cfg.USE_LEARNED_EDGE_BIAS and gt["und_u"].numel()>0:
                eu,ev=gt["und_u"],gt["und_v"]
                bias_vec=torch.nn.functional.softplus(pol.enc.bias_head(torch.cat([h[eu],h[ev],gt["und_attr"]],dim=-1))).squeeze(-1)*1e-3
            else:
                bias_vec=torch.zeros(gt["und_u"].numel(),device=dev)
            e2i=gt["edge_to_idx"]
            _,_,sol,gwl,isll=pol.rollout(inst,h,gw_emb,bias_vec,e2i,greedy=greedy)
        # convert isll dict -> isl_load array for next step
        isl_load=np.zeros(inst.n_edges)
        for ek,v in (isll.items() if hasattr(isll,'items') else []):
            idx=e2i.get(ek); 
            if idx is not None: isl_load[idx]=v
        sr=MP.compute_reward(inst,sol,gwl,isll)[1]['served_ratio']
        srs.append(sr)
    return srs

# find the encoder attribute name on the policy
enc_attr=[a for a in dir(pol) if 'enc' in a.lower() and not a.startswith('__')]
print("encoder attrs on policy:", enc_attr, flush=True)

# encoder is pol.enc (confirmed). Run the UNROLL SNR gate.
print("=== UNROLL SNR GATE (does SR climb across K steps?) ===", flush=True)
srs=unrolled_rollout(pol, inst, K=5, greedy=True)
for k,sr in enumerate(srs):
    print("  step %d: SR=%.4f" % (k+1, sr), flush=True)
print("  greedy baseline=0.763  dual=0.918", flush=True)
print("  step1→stepK delta = %+.4f" % (srs[-1]-srs[0]), flush=True)
