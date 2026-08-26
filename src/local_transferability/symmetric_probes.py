"""Strictly billed symmetric inner probes and equal-budget controls."""
from __future__ import annotations
import numpy as np

def symmetric_inner_probes(rho: float) -> np.ndarray:
    if rho <= 0: raise ValueError("rho must be positive")
    d=1/(2.0*np.sqrt(2.0))
    # Two shells are necessary: a single-radius circle makes the intercept
    # collinear with u1^2 + u2^2 and cannot identify a complete quadratic.
    return rho*np.array([[1,0],[-1,0],[0,1],[0,-1],[d,d],[-d,-d],[d,-d],[-d,d]],float)

def random_inner_probes(rho: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-rho,rho,size=(8,2))

def outer_more_points(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1,1,size=(8,2))
