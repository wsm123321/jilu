"""Seed-cluster bootstrap for selective-risk upper confidence bounds."""
from __future__ import annotations
import numpy as np

def selective_risk_by_cluster(seed_ids,accepted,reliable):
 seeds=np.asarray(seed_ids);a=np.asarray(accepted,bool);r=np.asarray(reliable,bool);unique=np.unique(seeds);out=[]
 for seed in unique:
  mask=(seeds==seed)&a;out.append((int(mask.sum()),int((mask&(~r)).sum())))
 return unique,np.asarray(out,int)
def cluster_bootstrap_risk_ucb(seed_ids,accepted,reliable,*,bootstrap_seed=0,draws=2000,quantile=.95):
 unique,counts=selective_risk_by_cluster(seed_ids,accepted,reliable);total=counts[:,0].sum();point=counts[:,1].sum()/total if total else float('nan')
 if total==0:return point,float('nan')
 rng=np.random.default_rng(bootstrap_seed);risks=[]
 for _ in range(draws):
  sampled=counts[rng.integers(0,len(unique),len(unique))];den=sampled[:,0].sum();risks.append(sampled[:,1].sum()/den if den else np.nan)
 return point,float(np.nanquantile(risks,quantile))
