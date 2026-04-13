
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def build_component_basis_U(
    placement_df_set: Set[int],
    model: Any,
    zero_flow_tol: float = 1e-6,
) -> Tuple[NDArray[np.float64], List[Set[int]], int]:
    n_bus: int = model.n_bus
    n: int = model.n                   
    n_l: int = model.n_l
    slack: int = model.slack_bus_idx

    if n_bus == 0:
        raise ValueError("Model has no buses.")

                                                                        
                                                                       
                                                                       
                                                
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        d0 = np.asarray(model.A @ model.theta0, dtype=np.float64)          
    if not np.isfinite(d0).all():
        bad = int(np.size(d0) - np.isfinite(d0).sum())
        logger.warning(
            "component_basis: detected %d non-finite d0 entries; "
            "clamping to zero for robust NDF/DF split.",
            bad,
        )
        d0 = np.nan_to_num(d0, nan=0.0, posinf=0.0, neginf=0.0)

                                                                        
                                                            
                                                                        
                                                                         
                                                                   
                                                                        
    ndf_adj: Dict[int, Set[int]] = defaultdict(set)
    ext2int = model._ext2int_map
    active_branches = model._active_branches
    _BR_FBUS = 0
    _BR_TBUS = 1

    for l_idx in range(n_l):
        is_ndf = l_idx not in placement_df_set
        is_zero_flow_df = (
            l_idx in placement_df_set
            and np.abs(d0[l_idx]) < zero_flow_tol
        )
        if is_ndf or is_zero_flow_df:
            fbus_ext = int(active_branches[l_idx, _BR_FBUS])
            tbus_ext = int(active_branches[l_idx, _BR_TBUS])
            u = ext2int[fbus_ext]
            v = ext2int[tbus_ext]
            ndf_adj[u].add(v)
            ndf_adj[v].add(u)

                                                                        
                                                         
                                                                        
                                                                     
                                        
    visited: Set[int] = set()
    all_components: List[Set[int]] = []

    for bus in range(n_bus):
        if bus in visited:
            continue
        component: Set[int] = set()
        queue: deque[int] = deque([bus])
        visited.add(bus)
        while queue:
            node = queue.popleft()
            component.add(node)
            for nb in ndf_adj.get(node, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        all_components.append(component)

                                                                        
                                                 
                                                                        
    slack_comp_idx: Optional[int] = None
    for i, comp in enumerate(all_components):
        if slack in comp:
            slack_comp_idx = i
            break

    if slack_comp_idx is None:
                                                                         
                                                                     
        raise ValueError(
            f"Slack bus (internal index {slack}) not found in any "
            f"connected component.  This indicates a bug."
        )

                                                        
    components: List[Set[int]] = [all_components[slack_comp_idx]]
    for i, comp in enumerate(all_components):
        if i != slack_comp_idx:
            components.append(comp)

    C: int = len(components)

                                                                        
                        
                                                                        
    s: int = C - 1

                                                                        
                             
                                                                        
                                                                     
                                                                      
                                           
     
                                                                      
                                                                        
                    
    def bus_to_state_idx(bus: int) -> Optional[int]:
        if bus == slack:
            return None
        return bus if bus < slack else bus - 1

    U = np.zeros((n, s), dtype=np.float64)

    for j in range(s):
        comp = components[j + 1]                                      
        for bus in comp:
            state_idx = bus_to_state_idx(bus)
            if state_idx is not None:
                U[state_idx, j] = 1.0

    return U, components, s


def rank_of_hidden_space(
    placement_df_set: Set[int],
    model: Any,
) -> int:
    _, _, s = build_component_basis_U(placement_df_set, model)
    return s


def validate_basis(
    U: NDArray[np.float64],
    placement_df_set: Set[int],
    model: Any,
    tol: float = 1e-12,
) -> bool:
    if U.shape[1] == 0:
        return True                                         

                                                                   
    ndf_indices = [l for l in range(model.n_l) if l not in placement_df_set]

    if len(ndf_indices) == 0:
                                                           
        return True

    A_ndf = model.A[ndf_indices, :]                      
    b_ndf = model.b0[ndf_indices]                      
    H_f_ndf = np.diag(b_ndf) @ A_ndf                    

    residual = H_f_ndf @ U                               
    return bool(np.max(np.abs(residual)) < tol)
