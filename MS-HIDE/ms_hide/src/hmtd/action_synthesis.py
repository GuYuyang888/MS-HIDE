
from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np
from numpy.typing import NDArray

from ..power.dc_model import DCModel
from .susceptance_update import compute_susceptance, verify_hiddenness


def synthesize_hidden_action(
    w: NDArray[np.float64],
    model: DCModel,
    placement_df_set: Set[int],
    U: NDArray[np.float64],
    cfg,
) -> Dict[str, Any]:
    w = np.asarray(w, dtype=np.float64).ravel()
                                              
    b_new, feasible = compute_susceptance(w, model, placement_df_set, U, cfg)

                                                                 
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        theta_shifted = model.theta0 + U @ w
    if not np.all(np.isfinite(theta_shifted)):
        theta_shifted = model.theta0.copy()

                                                                        
    if (not feasible) or (not np.all(np.isfinite(b_new))):
        return {
            "w": w.copy(),
            "b": b_new,
            "H_t": model.H0,
            "theta_shifted": theta_shifted,
            "feasible": False,
            "hidden": False,
            "hide_error": float("inf"),
        }

                                                          
    try:
        H_t = model.build_H(b_new)
    except ValueError:
        return {
            "w": w.copy(),
            "b": b_new,
            "H_t": model.H0,
            "theta_shifted": theta_shifted,
            "feasible": False,
            "hidden": False,
            "hide_error": float("inf"),
        }

                                     
    is_hidden, hide_error = verify_hiddenness(w, b_new, model, U, cfg)

    return {
        "w": w.copy(),
        "b": b_new,
        "H_t": H_t,
        "theta_shifted": theta_shifted,
        "feasible": feasible,
        "hidden": is_hidden,
        "hide_error": hide_error,
    }
