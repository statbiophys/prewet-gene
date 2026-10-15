"""Flory-Huggins bulk free energy of the continuum model (Fig. 5).

The Python counterpart of the Parameters and d2fb_dphi2 of src/bvp_solver.jl:
f_b(phi) = (1/nb) [phi log phi + (1 - phi) log(1 - phi)] + chi_b phi (1 - phi).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    nb: float = 1.0          # bulk molecule volume relative to solvent
    chi_b: float = 0.0       # Flory-Huggins parameter
    omega_b: float = 0.0     # surface field h0
    kappa: float = 1.0       # gradient-squared penalty
    phi_inf: float = 0.06    # far-field volume fraction
    L: float = 10.0          # domain size
    R: float = 3.0           # sphere radius
    eps: float = 1e-10       # floor of phi and 1 - phi


def d2fb_dphi2(phi, p):
    """Second derivative of f_b: sets the screening length, lambda = sqrt(f_b''/kappa)."""
    phi = np.clip(phi, p.eps, 1.0 - p.eps)
    return (1.0 / p.nb) * (1.0 / phi + 1.0 / (1 - phi)) - 2.0 * p.chi_b
