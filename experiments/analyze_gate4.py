"""Analyze frozen Gate 4 development and holdout evidence."""
from __future__ import annotations
import csv,gzip,hashlib,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib.pyplot as plt
import numpy as np
from local_transferability.clustered_risk import cluster_bootstrap_risk_ucb
OUT=ROOT/'results_step4';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def read(name):
 p=OUT/name
 if not p.exists() and (OUT/(name+'.gz')).exists():p=OUT/(name+'.gz')
 op=gzip.open if p.suffix=='.gz' else open
 with op(p,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize_classified(rows,split):
 groups=defaultdict(list)
 for r in rows:groups[(r['method'],r['family'],int(r['n_outer']),float(r['rho_inner']))].append(r)
 out=[]
 for key,g in sorted(groups.items()):
  accepted=[r['classification']=='Two-scale-stable-SPD' for r in g];reliable=[r['truth_stable']=='1' and r['center_reliable']=='1' for r in g];point,ucb=cluster_bootstrap_risk_ucb([int(r['seed']) for r in g],accepted,reliable,bootstrap_seed=41,draws=2000);scale_truth=[r for r in g if r['truth_stable']=='0'];detected=np.mean([r['classification']=='Scale-dependent' for r in scale_truth]) if scale_truth else np.nan
  out.append({'split':split,'method':key[0],'family':key[1],'n_outer':key[2],'rho_inner':key[3],'total':len(g),'intrinsic_accepted':sum(accepted),'coverage':np.mean(accepted),'selective_risk':point,'risk_ucb95':ucb,'scale_dependent_truth_count':len(scale_truth),'scale_dependent_detection':detected,'unidentifiable_rate':np.mean([r['classification']=='Unidentifiable' for r in g])})
 return out
def overall(rows,split):
 out=[]
 for method in sorted({r['method'] for r in rows}):
  g=[r for r in rows if r['method']==method];accepted=[r['classification']=='Two-scale-stable-SPD' for r in g];reliable=[r['truth_stable']=='1' and r['center_reliable']=='1' for r in g];point,ucb=cluster_bootstrap_risk_ucb([int(r['seed']) for r in g],accepted,reliable,bootstrap_seed=97,draws=2000);scale=[r for r in g if r['truth_stable']=='0']
  out.append({'split':split,'method':method,'total':len(g),'budget_mean':np.mean([int(r['total_budget']) for r in g]),'coverage':np.mean(accepted),'selective_risk':point,'risk_ucb95':ucb,'scale_detection':np.mean([r['classification']=='Scale-dependent' for r in scale]) if scale else np.nan,'unidentifiable_rate':np.mean([r['classification']=='Unidentifiable' for r in g]),'center_reliable_rate':np.mean([r['center_reliable']=='1' for r in g])})
 return out
def figures(overall_rows,detail):
 fig,ax=plt.subplots(figsize=(7,4.5));markers={'holdout':'o','unseen':'s'}
 for r in overall_rows:
  if not np.isnan(float(r['selective_risk'])):ax.scatter(float(r['coverage']),float(r['selective_risk']),s=70,marker=markers[r['split']],label=f"{r['split']}:{r['method']}")
 ax.axhline(.1,color='black',ls='--');ax.set(xlabel='Two-scale stable acceptance coverage',ylabel='Selective risk',xlim=(0,.25),ylim=(0,1));ax.legend(fontsize=7);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'figure1_semantic_risk_coverage.png',dpi=180);plt.close(fig)
 structured=[r for r in detail if r['method']=='Structured-Two-Scale'];families=sorted({r['family'] for r in structured});x=np.arange(len(families));fig,ax=plt.subplots(figsize=(10,4.5));vals=[];det=[]
 for f in families:
  g=[r for r in structured if r['family']==f];vals.append(np.mean([float(r['coverage']) for r in g]));finite=[float(r['scale_dependent_detection']) for r in g if not np.isnan(float(r['scale_dependent_detection']))];det.append(np.mean(finite) if finite else np.nan)
 ax.bar(x-.18,vals,.36,label='stable coverage');ax.bar(x+.18,det,.36,label='scale-dependent detection');ax.set_xticks(x,families,rotation=25,ha='right');ax.set_ylim(0,1);ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIG/'figure2_family_generalization.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4.5));
 for split in ('holdout','unseen'):
  g=[r for r in overall_rows if r['split']==split];ax.plot([float(r['budget_mean']) for r in g],[float(r['coverage']) for r in g],marker=markers[split],ls='',label=split)
 ax.set(xlabel='Mean billed evaluations',ylabel='Safe semantic coverage');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'figure3_cost_coverage.png',dpi=180);plt.close(fig)

def main():
 h=read('holdout_classified.csv');u=read('unseen_classified.csv');detail=summarize_classified(h,'holdout')+summarize_classified(u,'unseen');ov=overall(h,'holdout')+overall(u,'unseen');write('semantic_summary.csv',detail);write('cost_frontier.csv',ov);figures(ov,detail);print(json.dumps({'detail':len(detail),'overall':len(ov)}))
if __name__=='__main__':main()
