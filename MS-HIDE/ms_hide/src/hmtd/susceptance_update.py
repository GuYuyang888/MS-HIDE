
from __future__ import annotations

from typing import Set, Tuple

import numpy as np
from numpy.typing import NDArray

from ..power.dc_model import DCModel


                                                                        
            
                                                                        


def compute_susceptance(
    w: NDArray[np.float64],
    model: DCModel,
    placement_df_set: Set[int],
    U: NDArray[np.float64],
    cfg,
) -> Tuple[NDArray[np.float64], bool]:
    w = np.asarray(w, dtype=np.float64).ravel()
    eta_x: float = cfg.eta_x
    eps_denom: float = cfg.eps_denom

    A = model.A                     
    b0 = model.b0                 
    theta0 = model.theta0       

    if w.size != U.shape[1]:
        raise ValueError(
            f"w has size {w.size}, expected {U.shape[1]}."
        )
    if not np.all(np.isfinite(w)):
        return b0.copy(), False

                                                                            
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        d0 = A @ theta0                                      
    if not np.all(np.isfinite(d0)):
        return b0.copy(), False

                                                                            
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        theta_shifted = theta0 + U @ w                     
    if not np.all(np.isfinite(theta_shifted)):
        return b0.copy(), False
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        d_w = A @ theta_shifted                              
    if not np.all(np.isfinite(d_w)):
        return b0.copy(), False

                                    
    b_new = b0.copy()

    feasible = True

    for l_idx in placement_df_set:
                                                                           
                                                                        
                                                                        
                                                                           
        if np.abs(d0[l_idx]) < eps_denom:
                                                                     
            continue

                                 
        if np.abs(d_w[l_idx]) < eps_denom:
            feasible = False
                                                                  
            continue

                                           
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            b_candidate = b0[l_idx] * d0[l_idx] / d_w[l_idx]
        if not np.isfinite(b_candidate):
            feasible = False
            continue
        b_new[l_idx] = b_candidate

                               
        b_min = b0[l_idx] / (1.0 + eta_x)
        b_max = b0[l_idx] / (1.0 - eta_x)

                                                               
        if b_min > b_max:
            b_min, b_max = b_max, b_min

                                 
        if not (b_min <= b_new[l_idx] <= b_max):
            feasible = False

    if not np.all(np.isfinite(b_new)):
        feasible = False
        bad = ~np.isfinite(b_new)
        b_new[bad] = b0[bad]

    return b_new, feasible


def verify_hiddenness(
    w: NDArray[np.float64],
    b_new: NDArray[np.float64],
    model: DCModel,
    U: NDArray[np.float64],
    cfg,
) -> Tuple[bool, float]:
    hide_tol: float = cfg.hide_tol

    w = np.asarray(w, dtype=np.float64).ravel()
    b_new = np.asarray(b_new, dtype=np.float64).ravel()
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b_new)):
        return False, float("inf")
    if b_new.shape != model.b0.shape:
        return False, float("inf")

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        theta_shifted = model.theta0 + U @ w                
    if not np.all(np.isfinite(theta_shifted)):
        return False, float("inf")

                                                    
    try:
        H_new = model.build_H(b_new)                          
    except ValueError:
        return False, float("inf")

                          
    z_nom = model.z_nom                                  

                                       
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        delta_z_hide = H_new @ theta_shifted - z_nom         
    if not np.all(np.isfinite(delta_z_hide)):
        return False, float("inf")

    max_violation: float = float(np.max(np.abs(delta_z_hide)))
    is_hidden: bool = max_violation <= hide_tol

    return is_hidden, max_violation
