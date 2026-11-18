#!/usr/bin/env python3
# Isolate TRUE inference time: RL policy forward pass vs dual full re-solve vs greedy.
import time, numpy as np, torch
import hypatia_pop_instance as HP
import neural_solver_MCpath as MP
import unroll_solver as U
import gdh_dpq_v6e as DUAL
dev='cpu'; K=4

try: cfg=MP.default_cfg() if hasattr(MP,'default_cfg') else MP.CFG
except: import train_unroll_vr as TR; cfg=TR.cfg
pol=MP.ConstructionPolicy(cfg).to(dev)
sd=torch.load('policy_unroll.pt',map_location=dev)
import torch.nn as _nn
ck=sd['enc.node_in.weight'].shape[1]; cur=pol.enc.node_in
if cur.in_features!=ck:
    nl=_nn.Linear(ck,cur.out_features,bias=(cur.bias is not None)).to(dev)
    with torch.no_grad():
        m=min(cur.in_features,ck); nl.weight[:,:m].copy_(cur.weight[:,:m])
        if cur.bias is not None: nl.bias.copy_(cur.bias)
    pol.enc.node_in=nl
pol.load_state_dict(sd,strict=False); pol=U.widen_encoder(pol,dev); pol.eval()

inst,DST,GWS=HP.build_instance(scale=1.0,hotspot=True,use_std_gw=True)

# --- 1. TRUE RL inference: just the policy forward pass, timed in isolation ---
# build the graph features ONCE (this is preprocessing, not inference), then time forward passes
import torch as T
reps=50
# warm up
with T.no_grad():
    for _ in range(3):
        srs,fs,_=U.unrolled(pol,inst,K=K,greedy=True)
# time the FULL unrolled solve (what actually produces routes)
t=[]
with T.no_grad():
    for _ in range(5):
        t0=time.time(); U.unrolled(pol,inst,K=K,greedy=True); t.append(time.time()-t0)
print("RL unrolled() full solve: %.3f +/- %.3f s"%(np.mean(t),np.std(t)),flush=True)

# time JUST one policy forward pass (encoder + one decode step) if separable
# try to time pol.enc on the built graph
try:
    gt=U.build_gt8(inst) if hasattr(U,'build_gt8') else None
    if gt is not None:
        tf=[]
        with T.no_grad():
            for _ in range(reps):
                t0=time.time(); pol.enc(gt); tf.append(time.time()-t0)
        print("RL encoder forward pass ONLY: %.2f +/- %.2f ms"%(np.mean(tf)*1000,np.std(tf)*1000),flush=True)
except Exception as e:
    print("(encoder-only timing skipped: %s)"%str(e)[:50],flush=True)

# --- 2. dual full re-solve ---
t0=time.time(); DUAL.run_gdh_v6e(inst,inst.DST,inst.GW_SERVES,use_lambda=True,use_nu=True)
print("dual full re-solve: %.1f s"%(time.time()-t0),flush=True)

# --- 3. greedy ---
t0=time.time(); MP.greedy_nearest_baseline(inst)
print("greedy: %.3f s"%(time.time()-t0),flush=True)
print("[done]",flush=True)
