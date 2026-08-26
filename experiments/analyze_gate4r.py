"""Analyze corrected Gate 4-R evidence."""
from __future__ import annotations
import csv,gzip,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib.pyplot as plt
import numpy as np
from local_transferability.clustered_risk import cluster_bootstrap_risk_ucb
OUT=ROOT/'results_step4r';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def read(name):
 p=OUT/name
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def semantic_reliable(r):
 if r['method']=='Oracle-Scale':return r['center_semantically_valid']=='1'
 return r['center_semantically_valid']=='1' and r['outer_center_reliable']=='1' and r['inner_center_reliable']=='1'
def summarize(rows,split):
 out=[]
 for method in sorted({r['method'] for r in rows}):
  g=[r for r in rows if r['method']==method];accepted=[r['classification']=='Two-scale-stable-SPD' for r in g];good=[semantic_reliable(r) for r in g];risk,ucb=cluster_bootstrap_risk_ucb([int(r['seed']) for r in g],accepted,good,bootstrap_seed=77,draws=2000);scale=[r for r in g if r['scale_dependent_truth']=='1']
  out.append({'split':split,'method':method,'total':len(g),'mean_budget':np.mean([int(r['total_budget']) for r in g]),'coverage':np.mean(accepted),'selective_risk':risk,'risk_ucb95':ucb,'scale_detection':np.mean([r['classification']=='Scale-dependent' for r in scale]) if scale else np.nan,'unidentifiable_rate':np.mean([r['classification']=='Unidentifiable' for r in g]),'outer_reliable_rate':np.mean([r['outer_center_reliable']=='1' for r in g]),'inner_reliable_rate':np.mean([r['inner_center_reliable']=='1' for r in g])})
 return out
def family_summary(rows,split):
 out=[]
 for key in sorted({(r['method'],r['family']) for r in rows}):
  g=[r for r in rows if (r['method'],r['family'])==key];accepted=[r['classification']=='Two-scale-stable-SPD' for r in g];risk,ucb=cluster_bootstrap_risk_ucb([int(r['seed']) for r in g],accepted,[semantic_reliable(r) for r in g],bootstrap_seed=91,draws=2000);scale=[r for r in g if r['scale_dependent_truth']=='1']
  out.append({'split':split,'method':key[0],'family':key[1],'total':len(g),'coverage':np.mean(accepted),'selective_risk':risk,'risk_ucb95':ucb,'scale_detection':np.mean([r['classification']=='Scale-dependent' for r in scale]) if scale else np.nan,'outer_projection_bias_median':np.median([float(r['outer_projection_to_center']) for r in g]),'inner_projection_bias_median':np.median([float(r['inner_projection_to_center']) for r in g]),'projection_drift_median':np.median([float(r['outer_inner_projection_drift']) for r in g])})
 return out
def make_figures(overall,family):
 fig,ax=plt.subplots(figsize=(7,4.5))
 for r in overall:
  if not np.isnan(float(r['selective_risk'])):ax.scatter(float(r['coverage']),float(r['selective_risk']),s=70,label=f"{r['split']}:{r['method']}")
 ax.axhline(.1,color='black',ls='--');ax.set(xlabel='Corrected certificate coverage',ylabel='Selective risk',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'figure1_corrected_risk_coverage.png',dpi=180);plt.close(fig)
 structured=[r for r in family if r['method']=='Structured-Two-Scale'];families=sorted({r['family'] for r in structured});x=np.arange(len(families));cov=[];det=[]
 for f in families:
  g=[r for r in structured if r['family']==f];cov.append(np.mean([float(r['coverage']) for r in g]));finite=[float(r['scale_detection']) for r in g if not np.isnan(float(r['scale_detection']))];det.append(np.mean(finite) if finite else np.nan)
 fig,ax=plt.subplots(figsize=(10,4.5));ax.bar(x-.18,cov,.36,label='certificate coverage');ax.bar(x+.18,det,.36,label='scale-dependent detection');ax.set_xticks(x,families,rotation=25,ha='right');ax.set_ylim(0,1);ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIG/'figure2_corrected_family_results.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5));methods=sorted({r['method'] for r in overall});vals=[]
 for m in methods:vals.append(np.mean([float(r['coverage']) for r in overall if r['method']==m]))
 ax.bar(range(len(methods)),vals);ax.set_xticks(range(len(methods)),methods,rotation=25,ha='right');ax.set_ylabel('Mean corrected certificate coverage');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIG/'figure3_oracle_and_budget_comparison.png',dpi=180);plt.close(fig)
def main():
 h=read('holdout_classified.csv');u=read('unseen_classified.csv');overall=summarize(h,'holdout')+summarize(u,'unseen');family=family_summary(h,'holdout')+family_summary(u,'unseen');write('corrected_overall_summary.csv',overall);write('corrected_family_summary.csv',family);make_figures(overall,family);print(json.dumps({'overall':len(overall),'family':len(family)}))
if __name__=='__main__':main()
