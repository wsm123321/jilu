"""Multiscale curvature drift descriptors."""
from __future__ import annotations
import numpy as np

def vech(matrix) -> np.ndarray:
    K=np.asarray(matrix,float);return np.array([K[0,0],K[0,1],K[1,1]])

def normalized_scale_drift(outer,inner,epsilon=1e-12) -> float:
    Ko=np.asarray(outer,float);Ki=np.asarray(inner,float)
    return float(np.linalg.norm(Ko-Ki,'fro')/(.5*(np.linalg.norm(Ko,'fro')+np.linalg.norm(Ki,'fro'))+epsilon))

def covariance_standardized_drift(outer,inner,cov_outer,cov_inner) -> float:
    delta=vech(outer)-vech(inner);cov=np.asarray(cov_outer)+np.asarray(cov_inner)
    return float(delta@np.linalg.pinv(cov)@delta)
