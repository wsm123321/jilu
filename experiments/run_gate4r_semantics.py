"""Gate 4-R corrected development/holdout/unseen experiment."""
from __future__ import annotations
import argparse,csv,itertools,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,scipy
from numpy.polynomial.legendre import leggauss
from local_transferability.landscape_families import orthogonal_curvature,gradient_state,evaluate_family,center_hessian
from local_transferability.noise import stable_seed
from local_transferability.sampling import sample_design,sample_adaptive_landscape_trajectory
from local_transferability.symmetric_probes import symmetric_inner_probes,random_inner_probes,outer_more_points
from local_transferability.quadratic_estimator import fit_complete_quadratic,quadratic_design_matrix
from local_transferability.joint_uncertainty import joint_curvature_uncertainty
from local_transferability.scale_consistency import compare_scales
from local_transferability.identifiability_metrics import signed_spectrum_shape_error
from local_transferability.clustered_risk import cluster_bootstrap_risk_ucb
R=(1.,4.,16.);THETA=(0.,np.pi/4);DESIGNS=('sobol','uniform','trajectory');NO=(8,12);RHOI=(.25,.5);ETAS=(.01,.05);METHODS=('Residual-Only','Outer-More','Random-Inner','Structured-Two-Scale','Oracle-Scale')
THRESHOLD_GRID=list(itertools.product((.05,.10,.20,.30),(.10,.20,.30,.50),(.80,.90,.95),(.05,.10,.20,.30,.50)))
FAMILIES={'development':(('quadratic',0.),('axis_quartic',.05),('axis_quartic',.2),('axis_quartic',.5),('cross_quartic',.05),('cross_quartic',.2),('cross_quartic',.5)),'holdout':(('quadratic',0.),('axis_quartic',.05),('axis_quartic',.2),('axis_quartic',.5),('cross_quartic',.05),('cross_quartic',.2),('cross_quartic',.5)),'unseen':(('rotated_quartic',.05),('rotated_quartic',.2),('rotated_quartic',.5),('cubic',.05),('cubic',.2),('oscillatory',.05),('oscillatory',.2))}
def write(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluator(K,b,family,strength):return lambda U:evaluate_family(U,K,b,family=family,strength=strength)
def dense_projection(K,b,family,strength,rho):
 x,w=leggauss(10);U=np.array(np.meshgrid(x*rho,x*rho)).reshape(2,-1).T;weights=np.outer(w,w).ravel();X=quadratic_design_matrix(U);y=evaluate_family(U,K,b,family=family,strength=strength);coef=np.linalg.lstsq(X*np.sqrt(weights)[:,None],y*np.sqrt(weights),rcond=None)[0];return np.array([[coef[3],coef[4]],[coef[4],coef[5]]])
def relative_distance(A,B):return float(np.linalg.norm(A-B,'fro')/(np.linalg.norm(B,'fro')+1e-12))
def symmetric_drift(A,B):return float(np.linalg.norm(A-B,'fro')/(.5*(np.linalg.norm(A,'fro')+np.linalg.norm(B,'fro'))+1e-12))
def estimate_reliable(Khat,K0):
 if Khat is None:return False
 mag=abs(np.linalg.norm(Khat,'fro')-np.linalg.norm(K0,'fro'))/(np.linalg.norm(K0,'fro')+1e-12);spec=signed_spectrum_shape_error(K0,Khat);spd=np.min(np.linalg.eigvalsh(Khat))>1e-12
 return bool(mag<=.2 and spec<=.15 and spd)
def uncertainty_fields(prefix,u):
 return {f'{prefix}_estimable':int(u.available),f'{prefix}_relative_se':u.relative_magnitude_se if u.available else '',f'{prefix}_spectrum_width':u.spectrum_width if u.available else '',f'{prefix}_spd_probability':u.spd_probability if u.available else ''}
def generate(split,seeds):
 rows=[]
 for seed in seeds:
  ti=seed%2;theta=THETA[ti];nonstat=bool((seed//2)%2)
  for family,strength in FAMILIES[split]:
   for r in R:
    K=orthogonal_curvature(1,r,theta);K0=center_hessian(K,family,strength);b=gradient_state(1,nonstat);f=evaluator(K,b,family,strength)
    for eta in ETAS:
     rng=np.random.default_rng(stable_seed('g4r-noise',split,seed,family,strength,r,ti,nonstat,eta));outer_noise=rng.normal(0,eta,20);probe_noise=rng.normal(0,eta,32)
     for design in DESIGNS:
      pseed=stable_seed('g4r-points',split,seed,design);U20=sample_adaptive_landscape_trajectory(20,pseed,f,outer_noise) if design=='trajectory' else sample_design(design,20,pseed,K)
      outer_truth=(fit_complete_quadratic(U20,f(U20)).hessian if design=='trajectory' else dense_projection(K,b,family,strength,1.0));outer_truth_semantics='path-specific empirical projection' if design=='trajectory' else 'uniform-area projection'
      for no in NO:
       Uo=U20[:no];yo=f(Uo)+outer_noise[:no];eo=fit_complete_quadratic(Uo,yo);uo=joint_curvature_uncertainty(Uo,yo,eo,stable_seed('g4r-uo',split,seed,family,strength,r,ti,eta,design,no))
       for rhoi in RHOI:
        truth_inner_points=symmetric_inner_probes(rhoi);inner_truth=fit_complete_quadratic(truth_inner_points,f(truth_inner_points)).hessian;Bo=relative_distance(outer_truth,K0);Bi=relative_distance(inner_truth,K0);Do=symmetric_drift(outer_truth,inner_truth);center_valid=Bo<=.2 and Bi<=.2;scale_truth=Bo>.2 or Bi>.2
        active=('Structured-Two-Scale',) if split=='development' else METHODS
        for method in active:
         ui=None;scale=None;inner_est=None;outer_est=eo;outer_unc=uo;total=no
         if method=='Outer-More':
          extra=outer_more_points(stable_seed('g4r-outermore',split,seed,rhoi));U=np.vstack([Uo,extra]);y=np.r_[yo,f(extra)+probe_noise[:8]];outer_est=fit_complete_quadratic(U,y);outer_unc=joint_curvature_uncertainty(U,y,outer_est,stable_seed('g4r-uom',split,seed,rhoi));total=no+8
         elif method in ('Random-Inner','Structured-Two-Scale','Oracle-Scale'):
          Ui=symmetric_inner_probes(rhoi) if method!='Random-Inner' else random_inner_probes(rhoi,stable_seed('g4r-random',split,seed,rhoi));yi=f(Ui)+probe_noise[:8];inner_est=fit_complete_quadratic(Ui,yi);ui=joint_curvature_uncertainty(Ui,yi,inner_est,stable_seed('g4r-ui',split,seed,family,strength,r,ti,eta,design,no,rhoi,method));scale=compare_scales(eo,inner_est,uo,ui);total=no+8
         row={'split':split,'seed':seed,'family':family,'strength':strength,'r':r,'theta_index':ti,'nonstationary':int(nonstat),'eta':eta,'design':design,'n_outer':no,'rho_inner':rhoi,'method':method,'total_budget':total,'center_hessian_00':K0[0,0],'center_hessian_01':K0[0,1],'center_hessian_11':K0[1,1],'outer_truth_semantics':outer_truth_semantics,'outer_projection_to_center':Bo,'inner_projection_to_center':Bi,'outer_inner_projection_drift':Do,'center_semantically_valid':int(center_valid),'scale_dependent_truth':int(scale_truth),'outer_center_reliable':int(estimate_reliable(outer_est.hessian if outer_est.identifiable else None,K0)),'inner_center_reliable':int(estimate_reliable(inner_est.hessian if inner_est and inner_est.identifiable else None,K0)),'scale_available':int(scale.available) if scale else 0,'estimated_drift':scale.normalized_drift if scale and scale.available else '','T_diagnostic':scale.standardized_drift if scale and scale.available else ''}
         row.update(uncertainty_fields('outer',outer_unc));row.update(uncertainty_fields('inner',ui) if ui else {'inner_estimable':0,'inner_relative_se':'','inner_spectrum_width':'','inner_spd_probability':''});rows.append(row)
 return rows
def candidate_accept(r,t):
 se,sw,p,d=t
 return r['method']=='Structured-Two-Scale' and r['outer_estimable']==1 and r['inner_estimable']==1 and float(r['outer_relative_se'])<=se and float(r['inner_relative_se'])<=se and float(r['outer_spectrum_width'])<=sw and float(r['inner_spectrum_width'])<=sw and float(r['outer_spd_probability'])>=p and float(r['inner_spd_probability'])>=p and r['scale_available']==1 and float(r['estimated_drift'])<=d
def joint_reliable(r):return bool(r['center_semantically_valid'] and r['outer_center_reliable'] and r['inner_center_reliable'])
def freeze(rows):
 candidates=[]
 for t in THRESHOLD_GRID:
  accepted=[candidate_accept(r,t) for r in rows];selected=len({r['seed'] for r,a in zip(rows,accepted) if a})
  if selected<15:continue
  risk,ucb=cluster_bootstrap_risk_ucb([r['seed'] for r in rows],accepted,[joint_reliable(r) for r in rows],bootstrap_seed=stable_seed('g4r-bootstrap',t),draws=2000);coverage=np.mean(accepted);candidates.append({'relative_se':t[0],'spectrum_width':t[1],'spd_probability':t[2],'drift':t[3],'coverage':coverage,'risk':risk,'risk_ucb95':ucb,'selected_seed_clusters':selected})
 feasible=[x for x in candidates if x['risk_ucb95']<=.10];best=max(feasible,key=lambda x:(x['coverage'],-x['risk'],-x['relative_se'],-x['spectrum_width'],x['spd_probability'],-x['drift'])) if feasible else None;return candidates,best
def classify(rows,t):
 out=[]
 for r in rows:
  if r['method']=='Oracle-Scale':label='Two-scale-stable-SPD' if r['center_semantically_valid'] else 'Scale-dependent'
  elif r['method'] in ('Random-Inner','Structured-Two-Scale'):
   stats=r['outer_estimable']==1 and r['inner_estimable']==1 and float(r['outer_relative_se'])<=t['relative_se'] and float(r['inner_relative_se'])<=t['relative_se'] and float(r['outer_spectrum_width'])<=t['spectrum_width'] and float(r['inner_spectrum_width'])<=t['spectrum_width'] and float(r['outer_spd_probability'])>=t['spd_probability'] and float(r['inner_spd_probability'])>=t['spd_probability'] and r['scale_available']==1
   label='Unidentifiable' if not stats else ('Two-scale-stable-SPD' if float(r['estimated_drift'])<=t['drift'] else 'Scale-dependent')
  else:label='Unidentifiable'
  rr=dict(r);rr['classification']=label;rr['semantic_correct']=int((label=='Two-scale-stable-SPD' and joint_reliable(r)) or (label=='Scale-dependent' and r['scale_dependent_truth']));out.append(rr)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','holdout','unseen'],required=True);p.add_argument('--output',type=Path,default=Path('results_step4r'));p.add_argument('--overwrite',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=True);path=a.output/f'{a.split}_raw.csv'
 if path.exists() and not a.overwrite:raise SystemExit(f'Refusing to overwrite {path}')
 seeds=range(50) if a.split=='development' else (range(10000,10100) if a.split=='holdout' else range(20000,20100));rows=generate(a.split,seeds);write(path,rows);frozen=a.output/'frozen_thresholds.json'
 if a.split=='development':candidates,best=freeze(rows);write(a.output/'development_candidates.csv',candidates);frozen.write_text(json.dumps(best,indent=2),encoding='utf-8')
 else:
  if not frozen.exists():raise SystemExit('frozen thresholds required')
  best=json.loads(frozen.read_text());
  if best is None:raise SystemExit('development produced no safe threshold')
  write(a.output/f'{a.split}_classified.csv',classify(rows,best))
 manifest={'gate':'4-R','split':a.split,'git_commit_before_run':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'rows':len(rows),'seeds':[min(seeds),max(seeds)],'families':FAMILIES[a.split]};(a.output/f'{a.split}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'split':a.split,'rows':len(rows)}))
if __name__=='__main__':main()
