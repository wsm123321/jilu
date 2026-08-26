"""Gate 3B: noiseless quartic misspecification and projection bias."""
from __future__ import annotations
import argparse,csv,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np, scipy
from numpy.polynomial.legendre import leggauss
from local_transferability.landscape_families import orthogonal_curvature,gradient_state,evaluate_landscape
from local_transferability.noise import stable_seed
from local_transferability.sampling import sample_design,sample_observed_trajectory
from local_transferability.quadratic_estimator import fit_complete_quadratic
from local_transferability.variance_decomposition import dense_quadratic_projection,decompose_hessian_error
Q=(1.,); R=(1.,4.,16.); THETA=(0.,np.pi/4); DESIGNS=('sobol','uniform','trajectory'); NS=(8,12,20); BETA_RATIOS=(0.,.05,.20,.50); RHOS=(.25,.5,1.); SEEDS=range(50)

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def git_head():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 except:return 'unknown'
def weighted_projection(K,b,beta,rho):
 nodes,weights=leggauss(8); grid=np.array(np.meshgrid(nodes*rho,nodes*rho)).reshape(2,-1).T; w=np.outer(weights,weights).ravel()
 # weighted least squares by sqrt quadrature weights
 from local_transferability.quadratic_estimator import quadratic_design_matrix
 X=quadratic_design_matrix(grid); y=evaluate_landscape(grid,K,b,beta=beta); coef=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
 return np.array([[coef[3],coef[4]],[coef[4],coef[5]]])
def trajectory_projection(K,b,beta,rho,condition_key):
 pooled=[]
 for seed in range(500):
  U=sample_observed_trajectory(20,stable_seed('projection',condition_key,seed),K,b,np.zeros(20))*rho; pooled.append(U)
 points=np.vstack(pooled); return dense_quadratic_projection(points,evaluate_landscape(points,K,b,beta=beta))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results_step3'));p.add_argument('--overwrite',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=True);path=a.output/'misspecification_raw.csv'
 if path.exists() and not a.overwrite:raise SystemExit(f'Refusing to overwrite {path}')
 rows=[]; projections={}
 for q in Q:
  for r in R:
   for ti,theta in enumerate(THETA):
    K=orthogonal_curvature(q,r,theta)
    for nonstationary in (False,True):
     b=gradient_state(q,nonstationary)
     for beta_ratio in BETA_RATIOS:
      beta=beta_ratio*q
      for rho in RHOS:
       for design in DESIGNS:
        key=(q,r,ti,nonstationary,beta_ratio,rho,design)
        projection=weighted_projection(K,b,beta,rho) if design!='trajectory' else trajectory_projection(K,b,beta,rho,key)
        projections[key]=projection
        for seed in SEEDS:
         base_seed=stable_seed('misspec',design,seed)
         if design=='trajectory':U20=sample_observed_trajectory(20,base_seed,K,b,np.zeros(20))*rho
         else:U20=sample_design(design,20,base_seed,K)*rho
         for n in NS:
          U=U20[:n]; y=evaluate_landscape(U,K,b,beta=beta); est=fit_complete_quadratic(U,y)
          row={'q':q,'r':r,'theta_index':ti,'nonstationary':int(nonstationary),'beta_ratio':beta_ratio,'rho':rho,'design':design,'seed':seed,'n':n,'identifiable':int(est.identifiable),'condition':est.condition_number}
          if est.identifiable:
           row.update(decompose_hessian_error(est.hessian,projection,K))
          else:row.update({'finite_sample_to_projection':'','projection_to_center':'','total_to_center':''})
          rows.append(row)
 write_csv(path,rows);manifest={'git_commit_before_run':git_head(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'rows':len(rows),'projection':{'static':'8x8 Gauss-Legendre uniform-area square','trajectory':'pooled empirical 500 noiseless 20-point paths'}};(a.output/'misspecification_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':main()
