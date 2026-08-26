"""Orthogonalized quadratic and quartic landscape families for Gate 3."""
from __future__ import annotations
import numpy as np


def rotation(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def orthogonal_curvature(q: float, r: float, theta: float) -> np.ndarray:
    if q <= 0 or r < 1:
        raise ValueError("q must be positive and r must be at least one")
    Q = rotation(theta)
    spectrum = q * np.sqrt(2.0) * np.array([1.0, r]) / np.sqrt(1.0 + r * r)
    return Q @ np.diag(spectrum) @ Q.T


def gradient_state(q: float, nonstationary: bool) -> np.ndarray:
    return q * np.array([0.6, -0.4]) if nonstationary else np.zeros(2)


def center_hessian(base_curvature, family="quadratic", strength=0.0, phase=0.37, omega=5.3):
    """Exact Hessian at u=0 for every frozen landscape family."""
    K=np.asarray(base_curvature,dtype=float).copy()
    addition=np.zeros((2,2),dtype=float)
    if family in ("quadratic","axis_quartic","cross_quartic","rotated_quartic"):
        pass
    elif family=="cubic":
        addition[0,0]=6.0*0.23*strength
    elif family=="oscillatory":
        addition[0,0]=(omega**2)*np.cos(phase)*strength
    else:raise ValueError(f"unknown family: {family}")
    return K+addition


def evaluate_family(points, curvature, gradient=None, intercept=0.0, *, family="quadratic", strength=0.0, phase=0.37, omega=5.3):
    u=np.asarray(points,dtype=float);K=np.asarray(curvature,dtype=float);b=np.zeros(2) if gradient is None else np.asarray(gradient,dtype=float)
    base=intercept+u@b+0.5*np.einsum("ni,ij,nj->n",u,K,u)
    if family=="quadratic":extra=0.0
    elif family=="axis_quartic":extra=np.sum(u**4,axis=1)
    elif family=="cross_quartic":extra=u[:,0]**2*u[:,1]**2
    elif family=="rotated_quartic":
        direction=np.array([np.cos(phase),np.sin(phase)]);extra=(u@direction)**4
    elif family=="cubic":extra=(u[:,0]+0.23)**3-3*(0.23**2)*u[:,0]-0.23**3
    elif family=="oscillatory":extra=1-np.cos(omega*u[:,0]+phase)
    else:raise ValueError(f"unknown family: {family}")
    return base+strength*extra


def evaluate_landscape(points, curvature, gradient=None, intercept=0.0, beta=0.0):
    return evaluate_family(points,curvature,gradient,intercept,family="axis_quartic",strength=beta)
