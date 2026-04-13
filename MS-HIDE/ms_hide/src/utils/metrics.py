
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


                                                                    
                 
                                                                    

def detection_delay(
    bdd_flags: ArrayLike,
    true_attack_present: bool,
) -> Optional[int]:
    if not true_attack_present:
        return None
    flags = np.asarray(bdd_flags, dtype=bool)
    indices = np.nonzero(flags)[0]
    if indices.size == 0:
        return None
    return int(indices[0])


                                                                    
                        
                                                                    

def detection_success_rate(
    bdd_flags_list: Sequence[ArrayLike],
    attack_present_list: Sequence[bool],
) -> float:
    n_attack = 0
    n_detected = 0
    for flags, present in zip(bdd_flags_list, attack_present_list):
        if not present:
            continue
        n_attack += 1
        if np.any(np.asarray(flags, dtype=bool)):
            n_detected += 1
    if n_attack == 0:
        return 0.0
    return n_detected / n_attack


                                                                    
                         
                                                                    

def identification_accuracy(
    final_beliefs: Sequence[ArrayLike],
    true_hypotheses: Sequence[int],
) -> float:
    if len(final_beliefs) == 0:
        return 0.0
    n_correct = 0
    for belief, truth in zip(final_beliefs, true_hypotheses):
        b = np.asarray(belief, dtype=np.float64)
        if int(np.argmax(b)) == truth:
            n_correct += 1
    return n_correct / len(final_beliefs)


                                                                    
                           
                                                                    

def cumulative_actuation_cost(actuation_costs: ArrayLike) -> float:
    costs = np.asarray(actuation_costs, dtype=np.float64)
    return float(np.sum(costs))
