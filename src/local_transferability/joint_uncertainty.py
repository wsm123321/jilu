"""Joint curvature uncertainty from residual OLS covariance."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .uncertainty import estimate_uncertainty

@dataclass(frozen=True)
class JointCurvatureUncertainty:
    available: bool
    relative_magnitude_se: float | None
    spectrum_width: float | None
    spd_probability: float | None
    min_eigenvalue_q05: float | None
    covariance_hessian: np.ndarray | None
    draws: np.ndarray | None

def hessian_from_vech(v):return np.array([[v[0],v[1]],[v[1],v[2]]])
def signed_shape(K):
 e=np.sort(np.linalg.eigvalsh(K));norm=np.linalg.norm(e);return e/norm if norm>0 else np.array([np.nan,np.nan])
def joint_curvature_uncertainty(points,values,estimate,seed:int,draw_count:int=200,known_sigma=None):
 u=estimate_uncertainty(points,values,estimate,known_sigma=known_sigma)
 if u.coefficient_covariance is None or u.relative_magnitude_se is None or estimate.hessian is None:return JointCurvatureUncertainty(False,None,None,None,None,None,None)
 cov=u.coefficient_covariance[np.ix_([3,4,5],[3,4,5])];mean=np.array([estimate.hessian[0,0],estimate.hessian[0,1],estimate.hessian[1,1]])
 draws=np.random.default_rng(seed).multivariate_normal(mean,cov,size=draw_count);mats=np.array([hessian_from_vech(v) for v in draws]);shapes=np.array([signed_shape(K) for K in mats]);eig=np.linalg.eigvalsh(mats)
 width=float(np.max(np.quantile(shapes,.95,axis=0)-np.quantile(shapes,.05,axis=0)))
 return JointCurvatureUncertainty(True,u.relative_magnitude_se,width,float(np.mean(eig[:,0]>0)),float(np.quantile(eig[:,0],.05)),cov,draws)
