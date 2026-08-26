"""Gate 4 development/holdout/unseen multiscale semantic experiment."""
from __future__ import annotations
import argparse,csv,itertools,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np, scipy
from numpy.polynomial.legendre import leggauss
from local_transferability.landscape_families import orthogonal_curvature,gradient_state,evaluate_family
from local_transferability.noise import stable_seed
from local_transferability.sampling import sample_design,sample_adaptive_landscape_trajectory
from local_transferability.symmetric_probes import symmetric_inner_probes,random_inner_probes,outer_more_points
from local_transferability.quadratic_estimator import fit_complete_quadratic
from local_transferability.joint_uncertainty import joint_curvature_uncertainty
from local_transferability.scale_consistency import compare_scales
from local_transferability.identifiability_metrics import signed_spectrum_shape_error
from local_transferability.clustered_risk import cluster_bootstrap_risk_ucb

R=(1.,4.,16.);THETA=(0.,np.pi/4);DESIGNS=('sobol','uniform','trajectory');NO=(8,12);RHOI=(.25,.5);ETAS=(.01,.05);METHODS=('Residual-Only','Outer-More','Random-Inner','Structured-Two-Scale')
THRESHOLD_GRID=list(itertools.product((.05,.10,.20,.30),(.10,.20,.30,.50),(.80,.90,.95),(.05,.10,.20,.30,.50)))
FAMILIES={'development':(('quadratic',0.),('axis_quartic',.05),('axis_quartic',.2),('axis_quartic',.5),('cross_quartic',.05),('cross_quartic',.2),('cross_quartic',.5)),'holdout':(('quadratic',0.),('axis_quartic',.05),('axis_quartic',.2),('axis_quartic',.5),('cross_quartic',.05),('cross_quartic',.2),('cross_quartic',.5)),'unseen':(('rotated_quartic',.05),('rotated_quartic',.2),('rotated_quartic',.5),('cubic',.05),('cubic',.2),('oscillatory',.05),('oscillatory',.2))}

def write(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluator(K,b,family,strength):return lambda U:evaluate_family(U,K,b,family=family,strength=strength)
def dense_projection(K,b,family,strength,rho):
 x,w=leggauss(10);U=np.array(np.meshgrid(x*rho,x*rho)).reshape(2,-1).T;weights=np.outer(w,w).ravel();from local_transferability.quadratic_estimator import quadratic_design_matrix
 X=quadratic_design_matrix(U);y=evaluate_family(U,K,b,family=family,strength=strength);coef=np.linalg.lstsq(X*np.sqrt(weights)[:,None],y*np.sqrt(weights),rcond=None)[0];return np.array([[coef[3],coef[4]],[coef[4],coef[5]]])
def metrics(Khat,K0):
 mag=abs(np.linalg.norm(Khat,'fro')/np.sqrt(2)-1.0);spec=signed_spectrum_shape_error(K0,Khat);spd=np.min(np.linalg.eigvalsh(Khat))>1e-12;return mag,spec,spd

def generate(split,seeds):
 rows=[]
 for seed in seeds:
  # Prespecified balanced schedule avoids an unnecessary 4x Cartesian expansion.
  ti=seed % len(THETA);theta=THETA[ti];nonstat=bool((seed//len(THETA))%2)
  for family,strength in FAMILIES[split]:
   for r in R:
    K=orthogonal_curvature(1,r,theta);b=gradient_state(1,nonstat);f=evaluator(K,b,family,strength)
    for eta in ETAS:
       sigma=eta;noise_rng=np.random.default_rng(stable_seed('g4noise',split,seed,family,strength,r,ti,nonstat,eta));outer_noise=noise_rng.normal(0,sigma,20);probe_noise=noise_rng.normal(0,sigma,32)
       for design in DESIGNS:
        pseed=stable_seed('g4points',split,seed,design)
        U20=sample_adaptive_landscape_trajectory(20,pseed,f,outer_noise) if design=='trajectory' else sample_design(design,20,pseed,K)
        for no in NO:
         Uo=U20[:no];yo=f(Uo)+outer_noise[:no];eo=fit_complete_quadratic(Uo,yo);uo=joint_curvature_uncertainty(Uo,yo,eo,stable_seed('uo',split,seed,family,strength,r,ti,nonstat,eta,design,no));outer_truth=(fit_complete_quadratic(U20,f(U20)).hessian if design=='trajectory' else dense_projection(K,b,family,strength,1.0))
         for rhoi in RHOI:
          active_methods=('Structured-Two-Scale',) if split=='development' else METHODS;truth_inner_points=symmetric_inner_probes(rhoi);inner_truth=fit_complete_quadratic(truth_inner_points,f(truth_inner_points)).hessian;oracle_drift=np.linalg.norm(outer_truth-inner_truth,'fro')/(.5*(np.linalg.norm(outer_truth,'fro')+np.linalg.norm(inner_truth,'fro'))+1e-12);truth_stable=oracle_drift<=.2
          for method in active_methods:
           Ui=yi=ei=ui=scale=None;total=no
           if method=='Residual-Only': est=eo;unc=uo
           elif method=='Outer-More':
            extra=outer_more_points(stable_seed('outermore',split,seed,rhoi));U=np.vstack([Uo,extra]);y=np.r_[yo,f(extra)+probe_noise[:8]];est=fit_complete_quadratic(U,y);unc=joint_curvature_uncertainty(U,y,est,stable_seed('uom',split,seed,rhoi));total=no+8
           else:
            Ui=symmetric_inner_probes(rhoi) if method=='Structured-Two-Scale' else random_inner_probes(rhoi,stable_seed('randominner',split,seed,rhoi));yi=f(Ui)+probe_noise[:8];ei=fit_complete_quadratic(Ui,yi);ui=joint_curvature_uncertainty(Ui,yi,ei,stable_seed('ui',split,seed,family,strength,r,ti,nonstat,eta,design,no,rhoi,method));scale=compare_scales(eo,ei,uo,ui);est=ei;unc=ui;total=no+8
           mag=spec='';spd=False
           if est.identifiable and est.hessian is not None:mag,spec,spd=metrics(est.hessian,K)
           row={'split':split,'seed':seed,'family':family,'strength':strength,'r':r,'theta_index':ti,'nonstationary':int(nonstat),'eta':eta,'design':design,'n_outer':no,'rho_inner':rhoi,'method':method,'total_budget':total,'oracle_drift':oracle_drift,'truth_stable':int(truth_stable),'estimable':int(unc.available),'relative_se':unc.relative_magnitude_se if unc.available else '','spectrum_width':unc.spectrum_width if unc.available else '','spd_probability':unc.spd_probability if unc.available else '','scale_available':int(scale.available) if scale else 0,'drift':scale.normalized_drift if scale and scale.available else '','T_diagnostic':scale.standardized_drift if scale and scale.available else '','magnitude_error':mag,'spectrum_error':spec,'estimated_spd':int(spd),'center_reliable':int(mag!='' and mag<=.2 and spec<=.15 and spd)};rows.append(row)
 return rows

def candidate_accept(row,t):
 se,sw,p,d=t;return row['method']=='Structured-Two-Scale' and row['estimable']==1 and float(row['relative_se'])<=se and float(row['spectrum_width'])<=sw and float(row['spd_probability'])>=p and row['scale_available']==1 and float(row['drift'])<=d
def freeze(rows):
 candidates=[]
 for t in THRESHOLD_GRID:
  accepted=[candidate_accept(r,t) for r in rows];reliable=[bool(r['truth_stable'] and r['center_reliable']) for r in rows];selected_seeds=len({r['seed'] for r,a in zip(rows,accepted) if a})
  if selected_seeds<15:continue
  risk,ucb=cluster_bootstrap_risk_ucb([r['seed'] for r in rows],accepted,reliable,bootstrap_seed=stable_seed('g4bootstrap',t),draws=2000);coverage=np.mean(accepted)
  candidates.append({'relative_se':t[0],'spectrum_width':t[1],'spd_probability':t[2],'drift':t[3],'coverage':coverage,'risk':risk,'risk_ucb95':ucb,'selected_seed_clusters':selected_seeds})
 feasible=[x for x in candidates if x['risk_ucb95']<=.10];best=max(feasible,key=lambda x:(x['coverage'],-x['risk'],-x['relative_se'],-x['spectrum_width'],x['spd_probability'],-x['drift'])) if feasible else None
 return candidates,best

def classify(rows,t):
 out=[]
 for r in rows:
  estimable=r['method'] in ('Random-Inner','Structured-Two-Scale') and r['estimable']==1 and float(r['relative_se'])<=t['relative_se'] and float(r['spectrum_width'])<=t['spectrum_width'] and float(r['spd_probability'])>=t['spd_probability'] and r['scale_available']==1
  label='Unidentifiable' if not estimable else ('Two-scale-stable-SPD' if float(r['drift'])<=t['drift'] else 'Scale-dependent')
  rr=dict(r);rr['classification']=label;rr['semantic_correct']=int((label=='Two-scale-stable-SPD' and r['truth_stable'] and r['center_reliable']) or (label=='Scale-dependent' and not r['truth_stable']));out.append(rr)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','holdout','unseen'],required=True);p.add_argument('--output',type=Path,default=Path('results_step4'));p.add_argument('--overwrite',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=True);path=a.output/f'{a.split}_raw.csv'
 if path.exists() and not a.overwrite:raise SystemExit(f'Refusing to overwrite {path}')
 seeds=range(50) if a.split=='development' else (range(10000,10100) if a.split=='holdout' else range(20000,20100));rows=generate(a.split,seeds);write(path,rows);frozen=a.output/'frozen_thresholds.json'
 if a.split=='development':
  candidates,best=freeze(rows);write(a.output/'development_candidates.csv',candidates);frozen.write_text(json.dumps(best,indent=2),encoding='utf-8')
 else:
  if not frozen.exists():
   raise SystemExit('frozen thresholds required')
  best=json.loads(frozen.read_text())
  if best is None:
   raise SystemExit('development produced no safe threshold')
  write(a.output/f'{a.split}_classified.csv',classify(rows,best))
 manifest={'split':a.split,'git_commit_before_run':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'rows':len(rows),'seeds':[min(seeds),max(seeds)],'families':FAMILIES[a.split]};(a.output/f'{a.split}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'split':a.split,'rows':len(rows)}))
if __name__=='__main__':main()
