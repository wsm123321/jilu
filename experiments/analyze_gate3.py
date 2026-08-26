"""Create frozen Gate 3 summaries and figures from completed raw runs."""
from __future__ import annotations
import csv,hashlib,json,math,subprocess,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib.pyplot as plt
import numpy as np
OUT=ROOT/'results_step3'; FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)

def read(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8')))
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def wilson(success,total,z=1.96):
 if total==0:return (float('nan'),float('nan'))
 p=success/total;den=1+z*z/total;center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den;return center-half,center+half

def noise_summary(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['design'],int(r['n']),float(r['eta']))].append(r)
 result=[]
 for (design,n,eta),g in sorted(groups.items()):
  valid=[r for r in g if r['identifiable']=='1']; bad=sum(r['reliable']=='0' for r in valid); lo,hi=wilson(bad,len(valid))
  result.append({'design':design,'n':n,'eta':eta,'total':len(g),'rank_feasible':len(valid),'rank_coverage':len(valid)/len(g),'rank_risk':bad/len(valid) if valid else '', 'rank_risk_low':lo,'rank_risk_high':hi,'magnitude_error_median':np.median([float(r['magnitude_error']) for r in valid]) if valid else '','spectrum_error_median':np.median([float(r['signed_spectrum_error']) for r in valid]) if valid else '','spd_rate':np.mean([r['is_spd']=='1' for r in valid]) if valid else ''})
 return result

def misspec_summary(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['design'],float(r['rho']),float(r['beta_ratio']),int(r['n']))].append(r)
 result=[]
 for key,g in sorted(groups.items()):
  vals=lambda field:[float(r[field]) for r in g if r[field]!='']
  result.append({'design':key[0],'rho':key[1],'beta_ratio':key[2],'n':key[3],'runs':len(g),'projection_bias_median':np.median(vals('projection_to_center')),'finite_sample_error_median':np.median(vals('finite_sample_to_projection')),'total_center_error_median':np.median(vals('total_to_center')),'total_center_error_p90':np.quantile(vals('total_to_center'),.9)})
 return result

def combined_summary(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['scenario'],int(r['n']))].append(r)
 result=[]
 for key,g in sorted(groups.items()):
  accepted=[r for r in g if r['accepted_residual_gate']=='1']; failures=sum(r['reliable_center']=='0' for r in accepted);lo,hi=wilson(failures,len(accepted))
  result.append({'scenario':key[0],'n':key[1],'total':len(g),'accepted':len(accepted),'coverage':len(accepted)/len(g),'selective_risk':failures/len(accepted) if accepted else '','risk_low':lo,'risk_high':hi,'base_failure_rate':np.mean([r['reliable_center']=='0' for r in g])})
 return result

def figures(noise,misspec,risk,combined):
 designs=('sobol','uniform','trajectory');ns=(6,8,12,20);etas=(0,.01,.05,.1)
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,layout='constrained')
 for ax,design in zip(axes,designs):
  mat=np.full((len(etas),len(ns)),np.nan)
  for r in noise:
   if r['design']==design:mat[etas.index(r['eta']),ns.index(r['n'])]=r['rank_risk']
  im=ax.imshow(mat,vmin=0,vmax=1,cmap='magma',aspect='auto');ax.set_title(design);ax.set_xticks(range(4),ns);ax.set_yticks(range(4),etas);ax.set_xlabel('n')
 axes[0].set_ylabel('relative noise eta');fig.colorbar(im,ax=axes,label='Rank-only unreliable probability');fig.savefig(FIG/'figure1_noise_sample_risk.png',dpi=180);plt.close(fig)
 overall=[r for r in risk if r['group_field']=='overall'];fig,ax=plt.subplots(figsize=(7,4.5))
 for r in overall:
  if r['accepted']!='0':ax.scatter(float(r['coverage']),float(r['selective_risk']),s=70,label=r['gate'])
 ax.axhline(.1,color='black',ls='--',lw=1,label='10% target');ax.set(xlabel='Acceptance coverage',ylabel='Selective risk',xlim=(0,1),ylim=(0,.4));ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'figure2_holdout_risk_coverage.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,layout='constrained')
 for ax,design in zip(axes,designs):
  for beta in (0,.05,.2,.5):
   ys=[]
   for rho in (.25,.5,1.):ys.append(np.median([r['projection_bias_median'] for r in misspec if r['design']==design and r['beta_ratio']==beta and r['rho']==rho]))
   ax.plot((.25,.5,1.),ys,marker='o',label=f'beta/q={beta}');ax.set_title(design);ax.set_xlabel('region scale rho');ax.grid(alpha=.25)
 axes[0].set_ylabel('Projection-to-center relative bias');axes[-1].legend(fontsize=8);fig.savefig(FIG/'figure3_scale_misspecification_bias.png',dpi=180);plt.close(fig)
 scenarios=sorted({r['scenario'] for r in combined});x=np.arange(len(scenarios));coverage=[];riskvals=[]
 for s in scenarios:
  selected=[r for r in combined if r['scenario']==s];coverage.append(sum(r['accepted'] for r in selected)/sum(r['total'] for r in selected));acc=sum(r['accepted'] for r in selected);riskvals.append(sum(float(r['selective_risk'])*r['accepted'] for r in selected if r['selective_risk']!='')/acc if acc else np.nan)
 fig,ax=plt.subplots(figsize=(9,4.5));w=.36;ax.bar(x-w/2,coverage,w,label='coverage');ax.bar(x+w/2,riskvals,w,label='selective risk');ax.set_xticks(x,[s.replace('_','\n') for s in scenarios],fontsize=8);ax.set_ylim(0,1);ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIG/'figure4_combined_stress.png',dpi=180);plt.close(fig)

def main():
 hold=read('noise_holdout_raw.csv');miss=read('misspecification_raw.csv');risk=read('risk_coverage_holdout.csv');comb=read('combined_stress_raw.csv');n=noise_summary(hold);m=misspec_summary(miss);c=combined_summary(comb);write('noise_holdout_summary.csv',n);write('misspecification_summary.csv',m);write('combined_stress_summary.csv',c);figures(n,m,risk,c)
 files=sorted([p for p in OUT.rglob('*') if p.is_file() and p.name!='bundle_manifest.json']);manifest={'analysis_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':{str(p.relative_to(OUT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}};(OUT/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'noise_summary':len(n),'misspec_summary':len(m),'combined_summary':len(c),'files':len(files)}))
if __name__=='__main__':main()
