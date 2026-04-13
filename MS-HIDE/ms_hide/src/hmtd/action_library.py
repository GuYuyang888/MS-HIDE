
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import numpy as np
from numpy.typing import NDArray

from ..power.dc_model import DCModel
from ..power.dc_se import DCSE
from .action_synthesis import synthesize_hidden_action

logger = logging.getLogger(__name__)

                                          
_SCALE_SAMPLES: int = 5                                                       
_BISECT_ITERS: int = 40                                  
_INITIAL_SCALE: float = 1.0                                           
_NEAR_ZERO_TOL: float = 1e-10                                            
_DUP_TOL: float = 1e-8                                                  


class ActionLibrary:

    def __init__(
        self,
        model: DCModel,
        placement_df_set: Set[int],
        U: NDArray[np.float64],
        cfg,
    ) -> None:
        self._model = model
        self._placement_df_set = placement_df_set
        self._U = U
        self._cfg = cfg
                                                                      
        self._se_obs = DCSE(
            model.z_nom,
            sigma_rel=float(getattr(cfg, "sigma_rel", 0.01)),
            sigma_min=float(getattr(cfg, "sigma_min", 1e-4)),
        )
                                                                         
                                                                 
        self._df_indices = np.array(sorted(placement_df_set), dtype=np.int64)
        if self._df_indices.size > 0:
            b0_df = np.asarray(model.b0, dtype=np.float64)[self._df_indices]
            eta_x = float(getattr(cfg, "eta_x", 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                b_min = b0_df / (1.0 + eta_x)
                b_max = b0_df / (1.0 - eta_x)
            swap_mask = b_min > b_max
            if np.any(swap_mask):
                b_min_swapped = b_min.copy()
                b_min[swap_mask] = b_max[swap_mask]
                b_max[swap_mask] = b_min_swapped[swap_mask]
            span = b_max - b_min
            self._act_cost_ref = b0_df
            self._act_cost_scale = np.zeros_like(span)
            valid = span >= 1e-15
            self._act_cost_scale[valid] = 1.0 / span[valid]
            self._act_cost_valid = valid
        else:
            self._act_cost_ref = np.empty(0, dtype=np.float64)
            self._act_cost_scale = np.empty(0, dtype=np.float64)
            self._act_cost_valid = np.empty(0, dtype=bool)

        self.actions: List[Dict[str, Any]] = self._generate_library(
            model, placement_df_set, U, cfg
        )

                                                                        
                    
                                                                        

    @property
    def zero_action(self) -> Dict[str, Any]:
        return self.actions[0]

    def compute_actuation_cost(
        self, action: Dict[str, Any], cfg=None,
    ) -> float:
        if cfg is None:
            cfg = self._cfg

        b0 = self._model.b0
        b_new = action["b"]

        if self._df_indices.size == 0:
            return 0.0

        eta_x = float(getattr(cfg, "eta_x", self._cfg.eta_x))
        if eta_x == float(getattr(self._cfg, "eta_x", eta_x)):
            diff = np.asarray(b_new, dtype=np.float64)[self._df_indices] - self._act_cost_ref
            if not np.any(self._act_cost_valid):
                return 0.0
            scaled = diff[self._act_cost_valid] * self._act_cost_scale[self._act_cost_valid]
            return float(np.mean(scaled * scaled))

                                                              
        cost = 0.0
        valid_count = 0
        for l_idx in self._df_indices:
            b_min = b0[l_idx] / (1.0 + eta_x)
            b_max = b0[l_idx] / (1.0 - eta_x)
            if b_min > b_max:
                b_min, b_max = b_max, b_min
            span = b_max - b_min
            if span < 1e-15:
                continue
            cost += ((b_new[l_idx] - b0[l_idx]) / span) ** 2
            valid_count += 1

        return float(cost / valid_count) if valid_count > 0 else 0.0

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.actions[idx]

    def __repr__(self) -> str:
        return f"ActionLibrary(size={len(self.actions)})"

                                                                        
                                      
                                                                        

    def _generate_library(
        self,
        model: DCModel,
        placement_df_set: Set[int],
        U: NDArray[np.float64],
        cfg,
    ) -> List[Dict[str, Any]]:
        s = U.shape[1]                                  
        rng = np.random.default_rng(getattr(cfg, "seed", 1))
        max_actions: int = getattr(cfg, "max_actions", 15)
        num_dirs: int = getattr(cfg, "num_random_dirs", 30)

                                                                        
        w_zero = np.zeros(s, dtype=np.float64)
        zero_act = synthesize_hidden_action(
            w_zero, model, placement_df_set, U, cfg
        )
        zero_act["cost"] = self.compute_actuation_cost(zero_act, cfg)
        assert self._is_valid_action(zero_act, cfg, self._se_obs), (
            "Zero action must satisfy feasibility, hiddenness, and P1/P2."
        )

                                                                       
        directions: List[NDArray[np.float64]] = []
        for _ in range(num_dirs):
            q = rng.standard_normal(s)
            norm_q = np.linalg.norm(q)
            if norm_q < 1e-15:
                continue
            directions.append(q / norm_q)

                                                                   
        candidates: List[NDArray[np.float64]] = []
        for q in directions:
            for sign in (+1.0, -1.0):
                d = sign * q
                alpha_max = self._binary_search_scale(
                    d, model, placement_df_set, U, cfg, self._se_obs
                )
                if alpha_max < _NEAR_ZERO_TOL:
                    continue
                                                        
                for k in range(1, _SCALE_SAMPLES + 1):
                    alpha = alpha_max * k / _SCALE_SAMPLES
                    candidates.append(alpha * d)

                                                                     
        filtered: List[NDArray[np.float64]] = []
        for w_c in candidates:
            if np.linalg.norm(w_c, ord=np.inf) < _NEAR_ZERO_TOL:
                continue
            is_dup = False
            for w_f in filtered:
                if np.max(np.abs(w_c - w_f)) < _DUP_TOL:
                    is_dup = True
                    break
            if not is_dup:
                filtered.append(w_c)

        logger.info(
            "Action library: %d candidates after filtering (from %d raw).",
            len(filtered),
            len(candidates),
        )

                                                               
        if len(filtered) > 0:
            selected_ws = self._farthest_point_greedy(
                filtered, max_actions - 1                                  
            )
        else:
            selected_ws = []

                                                   
        actions: List[Dict[str, Any]] = [zero_act]
        for w_s in selected_ws:
            act = synthesize_hidden_action(
                w_s, model, placement_df_set, U, cfg
            )
                                                                                     
            if self._is_valid_action(act, cfg, self._se_obs):
                act["cost"] = self.compute_actuation_cost(act, cfg)
                actions.append(act)

        logger.info(
            "Action library finalised: %d actions (including zero action).",
            len(actions),
        )

        return actions

                                                                        
                                                                
                                                                        

    def _binary_search_scale(
        self,
        direction: NDArray[np.float64],
        model: DCModel,
        placement_df_set: Set[int],
        U: NDArray[np.float64],
        cfg,
        se_obs: DCSE,
    ) -> float:

        def _is_valid(alpha: float) -> bool:
            w = alpha * direction
            act = synthesize_hidden_action(
                w, model, placement_df_set, U, cfg
            )
            if not self._is_valid_action(act, cfg, se_obs):
                return False
            return True

                                                                 
        alpha_upper = _INITIAL_SCALE
        while _is_valid(alpha_upper) and alpha_upper < 1e6:
            alpha_upper *= 2.0

                                                                            
        if not _is_valid(alpha_upper / 2.0) and alpha_upper == _INITIAL_SCALE * 2.0:
                                                                              
                                                                               
            alpha_upper = _INITIAL_SCALE

                                    
        lo = 0.0
        hi = alpha_upper
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if _is_valid(mid):
                lo = mid
            else:
                hi = mid

        return lo

    @staticmethod
    def _is_valid_action(
        action: Dict[str, Any],
        cfg,
        se_obs: Any = None,
    ) -> bool:
        if not action.get("feasible", False):
            return False
        if not action.get("hidden", False):
            return False

        if se_obs is None:
            return True

        H_t = action.get("H_t")
        if H_t is None:
            return False
        obs = se_obs.check_observability(
            H_t,
            tau_obs=float(getattr(cfg, "tau_obs", 1e-12)),
        )
        return bool(obs.is_observable)

                                                                        
                                                 
                                                                        

    @staticmethod
    def _farthest_point_greedy(
        candidates: List[NDArray[np.float64]],
        max_actions: int,
    ) -> List[NDArray[np.float64]]:
        if len(candidates) == 0:
            return []

        n_pick = min(max_actions, len(candidates))

                                                                           
        C = np.array(candidates)          
        N = C.shape[0]

                                                     
        selected_mask = np.zeros(N, dtype=bool)

                                                                         
        norms = np.linalg.norm(C, axis=1)
        first_idx = int(np.argmax(norms))

        selected_indices: List[int] = [first_idx]
        selected_mask[first_idx] = True

                                                                               
        min_dist = np.full(N, np.inf)
                                                          
        diffs = C - C[first_idx]
        dists = np.linalg.norm(diffs, axis=1)
        min_dist = np.minimum(min_dist, dists)

        for _ in range(n_pick - 1):
                                                  
            min_dist_masked = min_dist.copy()
            min_dist_masked[selected_mask] = -1.0

                                                                  
            next_idx = int(np.argmax(min_dist_masked))
            if min_dist_masked[next_idx] <= 0.0:
                break                            

            selected_indices.append(next_idx)
            selected_mask[next_idx] = True

                                                           
            diffs = C - C[next_idx]
            dists = np.linalg.norm(diffs, axis=1)
            min_dist = np.minimum(min_dist, dists)

        return [candidates[i] for i in selected_indices]
