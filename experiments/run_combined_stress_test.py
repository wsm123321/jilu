"""Gate 3C: finite representative combinations, using frozen Gate 3A thresholds."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from local_transferability.landscape_families import orthogonal_curvature,gradient_state,evaluate_landscape
from local_transferability.noise import standard_noise,scaled_noise,stable_seed
from local_transferability.sampling import sample_design,sample_observed_trajectory
from local_transferability.quadratic_estimator import fit_complete_quadratic,standardized_design_condition
from local_transferability.identifiability_metrics import compute_identifiability_metrics
from local_transferability.uncertainty import estimate_uncertainty
from local_transferability.selective_gating import reliable
SCENARIOS=(('low_noise_weak_quartic',.01,.05,.5,False),('low_noise_strong_quartic',.01,.5,1.,False),('high_noise_weak_quartic',.10,.05,.5,False),('trajectory_nonstationary',.05,.20,.5,True))
R=(1.,4.,16.);THETA=(0.,np.pi/4);DESIGNS=('sobol','uniform','trajectory');NS=(8,12,20);SEEDS=range(20000,20100);PROBES=np.array([[-.91,-.37],[-.73,.58],[-.44,.19],[-.21,-.82],[.07,.31],[.29,-.54],[.48,.77],[.66,-.13],[.84,.42],[.95,-.69]])
def write(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results_step3'));p.add_argument('--overwrite',action='store_true');a=p.parse_args();path=a.output/'combined_stress_raw.csv'
 if path.exists() and not a.overwrite:raise SystemExit(f'Refusing to overwrite {path}')
 thresholds=json.loads((a.output/'frozen_thresholds.json').read_text());ru=thresholds['residual_uncertainty']['threshold'];rows=[];q=1.
 for scenario,eta,beta_ratio,rho,nonstationary in SCENARIOS:
  for design in DESIGNS:
   if scenario=='trajectory_nonstationary' and design!='trajectory':continue
   for seed in SEEDS:
    base=standard_noise(20,'combined',design,seed);noise=scaled_noise(base,eta,q,20);pointseed=stable_seed('combined-points',design,seed)
    for r in R:
     for ti,theta in enumerate(THETA):
      K=orthogonal_curvature(q,r,theta);b=gradient_state(q,nonstationary);beta=beta_ratio*q
      if design=='trajectory':U20=sample_observed_trajectory(20,pointseed,K,b,noise)*rho
      else:U20=sample_design(design,20,pointseed,K)*rho
      for n in NS:
       U=U20[:n];y=evaluate_landscape(U,K,b,beta=beta)+noise[:n];est=fit_complete_quadratic(U,y);_,cond,_=standardized_design_condition(U)
       row={'scenario':scenario,'design':design,'seed':seed,'r':r,'theta_index':ti,'n':n,'eta':eta,'beta_ratio':beta_ratio,'rho':rho,'nonstationary':int(nonstationary),'identifiable':int(est.identifiable),'standardized_condition':cond,'accepted_residual_gate':0,'reliable_center':0,'magnitude_error':'','signed_spectrum_error':'','is_spd':''}
       if est.identifiable:
        truthp=evaluate_landscape(PROBES,K,np.zeros(2));predp=evaluate_landscape(PROBES,est.hessian,np.zeros(2));m=compute_identifiability_metrics(K,est.hessian,truthp,predp);mag=abs(np.linalg.norm(est.hessian,'fro')/np.sqrt(2)-q)/q;good=reliable(mag,m.signed_spectrum_shape_error,m.is_spd);unc=estimate_uncertainty(U,y,est);accept=ru is not None and unc.relative_magnitude_se is not None and unc.relative_magnitude_se<=ru
        row.update({'accepted_residual_gate':int(accept),'reliable_center':int(good),'magnitude_error':mag,'signed_spectrum_error':m.signed_spectrum_shape_error,'is_spd':int(m.is_spd)})
       rows.append(row)
 a.output.mkdir(exist_ok=True);write(path,rows);manifest={'git_commit_before_run':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'rows':len(rows),'scenarios':SCENARIOS,'note':'Frozen residual gate from Gate 3A; no retuning under misspecification.'};(a.output/'combined_stress_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':main()
