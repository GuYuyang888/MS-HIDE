
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


                                                                    
           
                                                                    

def logsumexp(log_values: ArrayLike) -> float:
    a = np.asarray(log_values, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    a_max = np.max(a)
    if not np.isfinite(a_max):                                        
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


                                                                    
               
                                                                    

def log_normalize(
    log_weights: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    lw = np.asarray(log_weights, dtype=np.float64)
    log_z = logsumexp(lw)
    log_probs = lw - log_z
    probs = np.exp(log_probs)
                                              
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()                                                     
    return log_probs, probs


                                                                    
          
                                                                    

def safe_log(x: ArrayLike, floor: float = 1e-300) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.log(np.maximum(a, floor))


                                                                    
                    
                                                                    

def entropy_normalized(
    probs: ArrayLike,
    n_classes: int,
    eps_pi: float = 1e-12,
) -> float:
    if n_classes <= 1:
        return 0.0
    p = np.asarray(probs, dtype=np.float64)
                                                                 
    H = -np.sum(p * np.log(p + eps_pi))
    return float(H / np.log(n_classes))
