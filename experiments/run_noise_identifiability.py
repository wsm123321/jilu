"""Gate 3A: exact-quadratic noise stress and selective gating."""
from __future__ import annotations
import argparse,csv,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np, scipy
from local_transferability.landscape_families import orthogonal_curvature,gradient_state,evaluate_landscape
from local_transferability.noise import standard_noise,scaled_noise,stable_seed
from local_transferability.sampling import sample_design,sample_observed_trajectory
from local_transferability.quadratic_estimator import fit_complete_quadratic,standardized_design_condition
from local_transferability.identifiability_metrics import compute_identifiability_metrics
from local_transferability.uncertainty import estimate_uncertainty
from local_transferability.selective_gating import CONDITION_THRESHOLDS,UNCERTAINTY_THRESHOLDS,freeze_threshold,reliable,summarize_gate

Q=(.5,1.,2.); R=(1.,4.,16.); THETA=(0.,np.pi/8,np.pi/4); DESIGNS=('sobol','uniform','trajectory'); NS=(6,8,12,20); ETAS=(0.,.01,.05,.10); MAX_N=20
PROBES=np.array([[-.91,-.37],[-.73,.58],[-.44,.19],[-.21,-.82],[.07,.31],[.29,-.54],[.48,.77],[.66,-.13],[.84,.42],[.95,-.69]])

def values(U,K,b): return evaluate_landscape(U,K,b)
def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def git_head():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 except:return 'unknown'

def generate(split,seeds):
 rows=[]
 for design in DESIGNS:
  for seed in seeds:
   base_noise=standard_noise(MAX_N,split,design,seed)
   static_seed=stable_seed('points',split,design,seed)
   static_points=None if design=='trajectory' else sample_design(design,MAX_N,static_seed,np.eye(2))
   for q in Q:
    for r in R:
     for theta_index,theta in enumerate(THETA):
      K=orthogonal_curvature(q,r,theta)
      truth_probe=values(PROBES,K,np.zeros(2))
      for nonstationary in (False,True):
       b=gradient_state(q,nonstationary)
       for eta in ETAS:
        noise20=scaled_noise(base_noise,eta,q,MAX_N)
        if design=='trajectory':
         U20=sample_observed_trajectory(MAX_N,static_seed,K,b,noise20)
        else: U20=static_points
        for n in NS:
         U=U20[:n]; y=values(U,K,b)+noise20[:n]
         est=fit_complete_quadratic(U,y); srank,scond,_=standardized_design_condition(U)
         row={'split':split,'design':design,'seed':seed,'q':q,'r':r,'theta_index':theta_index,'nonstationary':int(nonstationary),'eta':eta,'n':n,'identifiable':int(est.identifiable),'standardized_rank':srank,'standardized_condition':scond,'raw_condition':est.condition_number,'magnitude_error':'','signed_spectrum_error':'','is_spd':'','inertia_mismatch':'','min_eigenvalue':'','pair_accuracy':'','pair_coverage':'','reliable':'','oracle_relative_se':'','residual_relative_se':'','residual_df':n-6}
         if est.identifiable and est.hessian is not None:
          pred_probe=values(PROBES,est.hessian,np.zeros(2)); m=compute_identifiability_metrics(K,est.hessian,truth_probe,pred_probe)
          magnitude=np.linalg.norm(est.hessian,'fro')/np.sqrt(2); magerr=abs(magnitude-q)/q
          oracle=estimate_uncertainty(U,y,est,known_sigma=eta*q); residual=estimate_uncertainty(U,y,est)
          good=reliable(magerr,m.signed_spectrum_shape_error,m.is_spd)
          row.update({'magnitude_error':magerr,'signed_spectrum_error':m.signed_spectrum_shape_error,'is_spd':int(m.is_spd),'inertia_mismatch':int(m.inertia_mismatch),'min_eigenvalue':m.estimated_min_eigenvalue,'pair_accuracy':m.pairwise_order_accuracy,'pair_coverage':m.pairwise_coverage,'reliable':int(good),'oracle_relative_se':oracle.relative_magnitude_se if oracle.relative_magnitude_se is not None else '','residual_relative_se':residual.relative_magnitude_se if residual.relative_magnitude_se is not None else ''})
         rows.append(row)
 return rows

def freeze(rows):
 valid=[r for r in rows if r['identifiable']==1]; good=[bool(int(r['reliable'])) for r in valid]
 thresholds={}
 for name,field,candidates in [('condition','standardized_condition',CONDITION_THRESHOLDS),('oracle_uncertainty','oracle_relative_se',UNCERTAINTY_THRESHOLDS),('residual_uncertainty','residual_relative_se',UNCERTAINTY_THRESHOLDS)]:
  vals=[float(r[field]) if r[field]!='' else np.nan for r in valid]; threshold,result=freeze_threshold(vals,good,candidates)
  thresholds[name]={'threshold':threshold,'development_coverage':None if result is None else result.coverage,'development_risk':None if result is None else result.selective_risk}
 return thresholds

def gate_rows(rows,thresholds):
 output=[]
 for gate in ('rank_only','condition','oracle_uncertainty','residual_uncertainty','combined'):
  accepted=[]; good=[]
  for r in rows:
   ok=bool(int(r['identifiable'])); cond=thresholds['condition']['threshold']; ou=thresholds['oracle_uncertainty']['threshold']; ru=thresholds['residual_uncertainty']['threshold']
   if gate=='condition': ok=ok and cond is not None and float(r['standardized_condition'])<=cond
   elif gate=='oracle_uncertainty': ok=ok and ou is not None and r['oracle_relative_se']!='' and float(r['oracle_relative_se'])<=ou
   elif gate=='residual_uncertainty': ok=ok and ru is not None and r['residual_relative_se']!='' and float(r['residual_relative_se'])<=ru
   elif gate=='combined': ok=ok and cond is not None and ru is not None and r['residual_relative_se']!='' and float(r['standardized_condition'])<=cond and float(r['residual_relative_se'])<=ru
   accepted.append(ok); good.append(bool(int(r['reliable'])) if r['reliable']!='' else False)
  s=summarize_gate(accepted,good); output.append({'gate':gate,'total':s.total,'accepted':s.accepted,'coverage':s.coverage,'selective_risk':s.selective_risk,'accepted_failure_mass':s.coverage*s.selective_risk if s.accepted else ''})
 return output

def main():
 p=argparse.ArgumentParser(); p.add_argument('--split',choices=['development','holdout'],required=True); p.add_argument('--output',type=Path,default=Path('results_step3')); p.add_argument('--overwrite',action='store_true'); a=p.parse_args(); a.output.mkdir(exist_ok=True)
 raw=a.output/f'noise_{a.split}_raw.csv'; frozen=a.output/'frozen_thresholds.json'
 if raw.exists() and not a.overwrite: raise SystemExit(f'Refusing to overwrite {raw}')
 seeds=range(50) if a.split=='development' else range(10000,10100); rows=generate(a.split,seeds); write_csv(raw,rows)
 if a.split=='development':
  thresholds=freeze(rows); frozen.write_text(json.dumps(thresholds,indent=2),encoding='utf-8')
 else:
  if not frozen.exists(): raise SystemExit('Development thresholds must be frozen first')
  thresholds=json.loads(frozen.read_text(encoding='utf-8')); write_csv(a.output/'risk_coverage_holdout.csv',gate_rows(rows,thresholds))
 manifest={'split':a.split,'git_commit_before_run':git_head(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'seeds':[min(seeds),max(seeds)],'rows':len(rows),'grid':{'q':Q,'r':R,'theta':THETA,'designs':DESIGNS,'n':NS,'eta':ETAS}}
 (a.output/f'noise_{a.split}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({'split':a.split,'rows':len(rows)}))
if __name__=='__main__':main()
