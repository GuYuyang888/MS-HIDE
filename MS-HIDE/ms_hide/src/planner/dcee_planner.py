
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..estimation.belief_update import BeliefUpdater
from ..estimation.residual_subspace import ResidualSubspace
from .exploit_score import (
    compute_detectability_normalized,
    compute_detectability_prob,
    compute_detectability_prob_batch,
    compute_detectability_weights,
    compute_g_norm_sq_vector,
    compute_gamma_th,
)
from .explore_score import (
    compute_support_entropy,
    compute_uncertainty,
    predict_uncertainty,
)
from .batch_rollout import (
    precompute_G_tensor,
    vectorized_horizon2_plan,
)

if TYPE_CHECKING:
    from ..attack.fdi_attack import FDIAttackModel
    from ..power.dc_se import DCSE

logger = logging.getLogger(__name__)


class DCEEPlanner:

    def __init__(
        self,
        attack_model: "FDIAttackModel",
        se: "DCSE",
        cfg,
    ) -> None:
        self.attack_model = attack_model
        self.se = se
        self.cfg = cfg
        self._belief_updater = BeliefUpdater(attack_model.n_hypotheses, cfg)

    def _exploit_gate_weight(self, belief, cfg) -> float:
        if not bool(getattr(cfg, "exploit_uncertainty_gate", False)):
            return 1.0

        kappa = max(float(getattr(cfg, "exploit_gate_strength", 0.0)), 0.0)
        floor = float(getattr(cfg, "exploit_gate_floor", 0.30))
        floor = float(np.clip(floor, 0.0, 1.0))
        if kappa <= 0.0:
            return 1.0

        h_supp = compute_support_entropy(belief, cfg)
        return float(np.clip(1.0 - kappa * h_supp, floor, 1.0))

                                                                        
                      
                                                                        

    def plan(
        self,
        belief,               
        action_library,                 
        cfg=None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Dict[str, Any], List[float]]:
        if cfg is None:
            cfg = self.cfg
        if rng is None:
            rng = np.random.default_rng()

        actions = action_library.actions
        if len(actions) == 0:
            logger.warning("Action library is empty; no action to select.")
            raise RuntimeError("Empty action library.")

        horizon: int = int(getattr(cfg, "horizon", 1))
        n_scenarios: int = int(getattr(cfg, "num_scenarios", 20))
        if horizon > 2:
            raise NotImplementedError(
                "horizon > 2 is not implemented; set cfg.horizon to 1 or 2."
            )

        try:
            if horizon <= 1:
                return self._plan_horizon_1(
                    belief, action_library, cfg, rng, n_scenarios
                )
            else:
                return self._plan_horizon_n(
                    belief, action_library, cfg, rng, n_scenarios, horizon
                )
        except Exception as exc:
            logger.error(
                "Planning failed (%s); falling back to zero action.", exc
            )
            return action_library.zero_action, [0.0] * len(actions)

                                                                        
                        
                                                                        

    def _plan_horizon_1(
        self,
        belief,
        action_library,
        cfg,
        rng: np.random.Generator,
        n_scenarios: int,
    ) -> Tuple[Dict[str, Any], List[float]]:
        actions = action_library.actions
        n_actions = len(actions)

                                                          
        g_dicts = self._compute_g_dicts(actions)
        g_norm_matrix = self._build_g_norm_matrix(actions, g_dicts)

                                                     
        nu = self._get_nu(g_dicts[0]) if g_dicts else None
        alpha_fa = float(getattr(cfg, "alpha_fa", 0.00256))
        steepness = float(getattr(cfg, "detect_steepness", 4.0))
        confidence_beta = max(
            float(getattr(cfg, "exploit_confidence_beta", 0.0)),
            0.0,
        )
        gamma_th_val = compute_gamma_th(nu, alpha_fa) if nu else 45.0
        d_prob_all = compute_detectability_prob_batch(
            belief,
            g_norm_matrix,
            gamma_th_val,
            steepness,
            confidence_beta=confidence_beta,
        )

                                                                          
        act_costs = self._compute_action_costs(actions, action_library, cfg)
        rho_u: float = float(getattr(cfg, "rho_u", 0.05))
        xi_weight: float = float(getattr(cfg, "xi_weight", 1.0))

        if getattr(cfg, "xi_adaptive", False):
            xi_weight = xi_weight * max(0.1, compute_support_entropy(belief, cfg))

        exploit_gate_w = self._exploit_gate_weight(belief, cfg)
        d_prob_all = d_prob_all * exploit_gate_w

                                                                      
        from .explore_score import predict_uncertainty
        crn_seed = int(rng.integers(0, 2**31))
        xi_all = np.empty(n_actions, dtype=np.float64)
        for i in range(n_actions):
            crn_rng = np.random.default_rng(crn_seed)
            xi_all[i] = predict_uncertainty(
                belief, g_dicts[i], cfg,
                belief_updater=self._belief_updater, rng=crn_rng,
            )

                                                               
        costs_arr = -d_prob_all + xi_weight * xi_all + rho_u * act_costs
        costs = costs_arr.tolist()

                                                     
        best_idx = int(np.argmin(costs_arr))

        logger.debug(
            "Horizon-1 plan: best_idx=%d, cost=%.6f (of %d actions).",
            best_idx,
            costs[best_idx],
            n_actions,
        )

        return actions[best_idx], costs

                                                                        
                                        
                                                                        

    def _plan_horizon_n(
        self,
        belief,
        action_library,
        cfg,
        rng: np.random.Generator,
        n_scenarios: int,
        horizon: int,
    ) -> Tuple[Dict[str, Any], List[float]]:
        actions = action_library.actions
        n_actions = len(actions)

                                             
        g_dicts = self._compute_g_dicts(actions)
        g_norm_matrix = self._build_g_norm_matrix(actions, g_dicts)
        act_costs = self._compute_action_costs(actions, action_library, cfg)

        rho_u: float = float(getattr(cfg, "rho_u", 0.05))
        xi_weight: float = float(getattr(cfg, "xi_weight", 1.0))
        eps_d: float = float(getattr(cfg, "eps_d", 1e-12))
        eps_prob: float = float(getattr(cfg, "eps_prob", 1e-12))
        q_c: float = float(getattr(cfg, "q_c", 1e-5))
        lambda_H: float = float(getattr(cfg, "lambda_H", 0.60))
        P0: float = float(getattr(cfg, "prior_c_var", 0.0025))
        steepness: float = float(getattr(cfg, "detect_steepness", 4.0))
        confidence_beta: float = max(
            float(getattr(cfg, "exploit_confidence_beta", 0.0)),
            0.0,
        )

                                                                               
        if getattr(cfg, "xi_adaptive", False):
            H_supp = compute_support_entropy(belief, cfg)
            xi_weight = xi_weight * max(0.1, H_supp)

                                   
        nu = self._get_nu(g_dicts[0]) if g_dicts else None
        if nu is None:
            logger.warning("No residual subspace; returning zero action.")
            return action_library.zero_action, [0.0] * n_actions

                                          
        alpha_fa = float(getattr(cfg, "alpha_fa", 0.00256))
        gamma_th_val = compute_gamma_th(nu, alpha_fa)

        n_hyp = self.attack_model.n_hypotheses

                                              
        G_all, g_norm_sq_all = precompute_G_tensor(g_dicts, n_hyp, nu)

                                
        best_first, marginal_costs = vectorized_horizon2_plan(
            belief_pi=np.asarray(belief.pi, dtype=np.float64),
            belief_mu=np.asarray(belief.mu, dtype=np.float64),
            belief_P=np.asarray(belief.P, dtype=np.float64),
            G_all=G_all,
            g_norm_sq_all=g_norm_sq_all,
            g_norm_matrix=g_norm_matrix,
            act_costs=act_costs,
            n_scenarios=n_scenarios,
            rng=rng,
            q_c=q_c,
            rho_u=rho_u,
            xi_weight=xi_weight,
            lambda_H=lambda_H,
            P0=P0,
            support_entropy_mode=str(
                getattr(cfg, "support_entropy_mode", "shannon")
            ),
            renyi_alpha=float(getattr(cfg, "renyi_alpha", 2.0)),
            renyi_entropy_blend=float(
                getattr(cfg, "renyi_entropy_blend", 0.5)
            ),
            eps_d=eps_d,
            eps_prob=eps_prob,
            gamma_th=gamma_th_val,
            steepness=steepness,
            exploit_confidence_beta=confidence_beta,
            terminal_xi_bonus=float(getattr(cfg, "terminal_xi_bonus", 1.0)),
            terminal_entropy_bonus=float(
                getattr(cfg, "terminal_entropy_bonus", 0.0)
            ),
            terminal_margin_bonus=float(
                getattr(cfg, "terminal_margin_bonus", 0.0)
            ),
            entropy_drop_bonus=float(getattr(cfg, "entropy_drop_bonus", 0.0)),
            cov_trace_enable=bool(getattr(cfg, "cov_trace_enable", False)),
            cov_trace_weight=float(getattr(cfg, "cov_trace_weight", 0.0)),
            cov_trace_terminal_bonus=float(
                getattr(cfg, "cov_trace_terminal_bonus", 0.0)
            ),
            info_gain_bonus=float(getattr(cfg, "info_gain_bonus", 0.0)),
            step1_xi_coupling=float(getattr(cfg, "step1_xi_coupling", 0.0)),
            step1_exploit_coupling=float(
                getattr(cfg, "step1_exploit_coupling", 0.0)
            ),
            step1_dual_shared_coupling=float(
                getattr(cfg, "step1_dual_shared_coupling", 0.0)
            ),
            step1_phase_dual_coupling=float(
                getattr(cfg, "step1_phase_dual_coupling", 0.0)
            ),
            step1_temporal_coupling=float(
                getattr(cfg, "step1_temporal_coupling", 0.0)
            ),
            step1_temporal_detect_ref=float(
                getattr(cfg, "step1_temporal_detect_ref", 0.55)
            ),
            step1_temporal_detect_floor=float(
                getattr(cfg, "step1_temporal_detect_floor", 0.0)
            ),
            topk_ratio=float(getattr(cfg, "rollout_topk_ratio", 0.20)),
            topk_min=int(getattr(cfg, "rollout_topk_min", 2)),
            topk_screen_scenarios=int(
                getattr(cfg, "rollout_topk_screen_scenarios", 1)
            ),
            exploit_uncertainty_gate=bool(
                getattr(cfg, "exploit_uncertainty_gate", False)
            ),
            exploit_gate_strength=float(
                getattr(cfg, "exploit_gate_strength", 0.0)
            ),
            exploit_gate_floor=float(getattr(cfg, "exploit_gate_floor", 0.30)),
        )

        logger.debug(
            "Horizon-%d plan (vectorized): best_first=%d, cost=%.6f "
            "(enumerated %d pairs over %d scenarios).",
            horizon,
            best_first,
            marginal_costs[best_first],
            n_actions * n_actions,
            n_scenarios,
        )

        return actions[best_first], marginal_costs

                                                                        
                             
                                                                        

    def _score_single_step(
        self,
        belief,
        action: Dict[str, Any],
        g_dict: Dict[int, NDArray[np.float64]],
        d_prob: float,
        action_library,
        cfg,
        rng: np.random.Generator,
        crn_seed: int,
        act_cost: Optional[float] = None,
    ) -> float:
        rho_u: float = float(getattr(cfg, "rho_u", 0.05))
        xi_weight: float = float(getattr(cfg, "xi_weight", 1.0))

                                                                               
        if getattr(cfg, "xi_adaptive", False):
            H_supp = compute_support_entropy(belief, cfg)
            xi_weight = xi_weight * max(0.1, H_supp)

                                                                               
        exploit_term = -self._exploit_gate_weight(belief, cfg) * d_prob

                                                      
                                                                                  
        crn_rng = np.random.default_rng(crn_seed)
        explore_term = predict_uncertainty(
            belief,
            g_dict,
            cfg,
            belief_updater=self._belief_updater,
            rng=crn_rng,
        )

                        
        if act_cost is None:
            act_cost = float(
                action.get("cost", action_library.compute_actuation_cost(action, cfg))
            )
        else:
            act_cost = float(act_cost)

        return exploit_term + xi_weight * explore_term + rho_u * act_cost

                                                                        
                               
                                                                        

    def _rollout_one_scenario(
        self,
        belief,
        action_indices_seq: List[int],
        actions: List[Dict[str, Any]],
        all_g_dicts: List[Dict[int, NDArray[np.float64]]],
        h_sample: int,
        c_sample: float,
        action_library,
        cfg,
        rng: np.random.Generator,
        g_norm_matrix: Optional[NDArray[np.float64]] = None,
        act_costs: Optional[NDArray[np.float64]] = None,
    ) -> float:
                                                            
        belief_sim = belief.copy() if hasattr(belief, "copy") else copy.deepcopy(belief)
        total_cost = 0.0
        rho_u: float = float(getattr(cfg, "rho_u", 0.05))
        xi_weight: float = float(getattr(cfg, "xi_weight", 1.0))
        eps_d: float = float(getattr(cfg, "eps_d", 1e-12))
        eps_prob: float = float(getattr(cfg, "eps_prob", 1e-12))
        entropy_drop_bonus: float = float(getattr(cfg, "entropy_drop_bonus", 0.0))
        terminal_margin_bonus: float = max(
            float(getattr(cfg, "terminal_margin_bonus", 0.0)), 0.0
        )
        info_gain_bonus: float = max(
            float(getattr(cfg, "info_gain_bonus", 0.0)), 0.0
        )
        step1_xi_coupling: float = max(
            float(getattr(cfg, "step1_xi_coupling", 0.0)), 0.0
        )
        step1_exploit_coupling: float = max(
            float(getattr(cfg, "step1_exploit_coupling", 0.0)), 0.0
        )
        step1_dual_shared_coupling: float = max(
            float(getattr(cfg, "step1_dual_shared_coupling", 0.0)), 0.0
        )
        step1_phase_dual_coupling: float = max(
            float(getattr(cfg, "step1_phase_dual_coupling", 0.0)), 0.0
        )
        step1_temporal_coupling: float = max(
            float(getattr(cfg, "step1_temporal_coupling", 0.0)), 0.0
        )
        step1_temporal_detect_ref: float = max(
            float(getattr(cfg, "step1_temporal_detect_ref", 0.55)),
            1e-6,
        )
        step1_temporal_detect_floor: float = float(
            np.clip(getattr(cfg, "step1_temporal_detect_floor", 0.0), 0.0, 1.0)
        )
        if step1_temporal_detect_floor >= step1_temporal_detect_ref:
            step1_temporal_detect_ref = step1_temporal_detect_floor + 1e-6
        exploit_floor = float(np.clip(getattr(cfg, "exploit_gate_floor", 0.30), 0.0, 1.0))
        prev_h_supp = None
        prev_d_prob = None

        n_steps = len(action_indices_seq)

        if g_norm_matrix is None:
            g_norm_matrix = self._build_g_norm_matrix(actions, all_g_dicts)
        if act_costs is None:
            act_costs = self._compute_action_costs(actions, action_library, cfg)

        for k in range(n_steps):
            a_idx = int(action_indices_seq[k])
            g_dict_k = all_g_dicts[a_idx]
            prior_pi = np.asarray(belief_sim.pi, dtype=np.float64)

                                                           
            nu_k = self._get_nu(g_dict_k)
            alpha_fa_k = float(getattr(cfg, "alpha_fa", 0.00256))
            steepness_k = float(getattr(cfg, "detect_steepness", 4.0))
            confidence_beta_k = max(
                float(getattr(cfg, "exploit_confidence_beta", 0.0)),
                0.0,
            )
            gamma_th_k = compute_gamma_th(nu_k, alpha_fa_k) if nu_k else 45.0
            d_prob_k = compute_detectability_prob(
                belief_sim,
                g_dict_k,
                gamma_th_k,
                steepness_k,
                confidence_beta=confidence_beta_k,
            )

                                      
            act_cost_k = float(act_costs[a_idx])

                                            
            nu = self._get_nu(g_dict_k)
            if nu is not None:
                epsilon = rng.standard_normal(nu)
                if h_sample == 0 or h_sample not in g_dict_k:
                    y_sim = epsilon
                else:
                    y_sim = g_dict_k[h_sample] * c_sample + epsilon
            else:
                                                  
                y_sim = None

                                     
            if y_sim is not None:
                try:
                    belief_pred = self._belief_updater.predict(belief_sim, cfg)
                    belief_sim = self._belief_updater.update(
                        belief_pred, y_sim, g_dict_k, cfg
                    )
                except Exception:
                    logger.debug(
                        "Rollout step %d: belief update failed, "
                        "skipping update.",
                        k,
                    )
            post_pi = np.asarray(belief_sim.pi, dtype=np.float64)
            info_gain = 0.0
            if info_gain_bonus > 0.0:
                p = np.maximum(post_pi, eps_prob)
                q = np.maximum(prior_pi, eps_prob)
                p /= max(float(np.sum(p)), eps_prob)
                q /= max(float(np.sum(q)), eps_prob)
                info_gain = float(np.sum(p * np.log(p / q)))

                                               
            xi_k, _, _ = compute_uncertainty(belief_sim, cfg)
            h_supp_k = compute_support_entropy(belief_sim, cfg)
            exploit_gate_w = self._exploit_gate_weight(belief_sim, cfg)
            if step1_exploit_coupling > 0.0 and prev_h_supp is not None:
                stage1_exploit_w = float(
                    np.clip(
                        1.0 - step1_exploit_coupling * float(prev_h_supp),
                        exploit_floor,
                        1.0,
                    )
                )
                exploit_gate_w *= stage1_exploit_w
            xi_stage_weight = xi_weight
            if step1_xi_coupling > 0.0 and prev_h_supp is not None:
                xi_stage_weight = xi_weight * (
                    1.0 + step1_xi_coupling * float(prev_h_supp)
                )
            if step1_dual_shared_coupling > 0.0 and prev_h_supp is not None:
                shared_gate = float(
                    np.clip(
                        1.0 - step1_dual_shared_coupling * float(prev_h_supp),
                        exploit_floor,
                        1.0,
                    )
                )
                exploit_gate_w *= shared_gate
                xi_stage_weight *= (2.0 - shared_gate)
            if (
                step1_phase_dual_coupling > 0.0
                and prev_h_supp is not None
                and prev_d_prob is not None
            ):
                                                                    
                                                                        
                                                                           
                                                                   
                phase_signal = float(prev_h_supp) * max(
                    1.0 - float(prev_d_prob), 0.0
                )
                phase_gate = float(
                    exploit_floor
                    + (1.0 - exploit_floor)
                    * np.exp(-step1_phase_dual_coupling * max(phase_signal, 0.0))
                )
                phase_gate = float(np.clip(phase_gate, exploit_floor, 1.0))
                exploit_gate_w *= phase_gate
                xi_stage_weight *= (2.0 - phase_gate) * 1.15
            if step1_temporal_coupling > 0.0 and prev_d_prob is not None:
                d_norm = max(
                    (float(prev_d_prob) - step1_temporal_detect_floor)
                    / (step1_temporal_detect_ref - step1_temporal_detect_floor),
                    0.0,
                )
                phase = float(
                    np.clip(
                        step1_temporal_coupling * d_norm,
                        0.0,
                        0.9,
                    )
                )
                exploit_gate_w *= (1.0 - phase)
                xi_stage_weight *= (1.0 + phase)

            step_cost = (
                -exploit_gate_w * d_prob_k + xi_stage_weight * xi_k + rho_u * act_cost_k
            )
            if info_gain_bonus > 0.0:
                step_cost -= xi_stage_weight * info_gain_bonus * info_gain
                                                                           
                                                                      
            if entropy_drop_bonus > 0.0 and prev_h_supp is not None:
                entropy_drop = max(float(prev_h_supp) - float(h_supp_k), 0.0)
                step_cost -= xi_stage_weight * entropy_drop_bonus * entropy_drop
            total_cost += step_cost
            prev_h_supp = h_supp_k
            prev_d_prob = d_prob_k

                                          
                                                                      
                                                                          
                                                                      
        terminal_bonus = float(getattr(cfg, "terminal_xi_bonus", 1.0))
        xi_terminal, _, _ = compute_uncertainty(belief_sim, cfg)
        total_cost += xi_weight * terminal_bonus * xi_terminal
        if terminal_margin_bonus > 0.0:
            pi_term = np.asarray(belief_sim.pi, dtype=np.float64)
            pi_term = np.nan_to_num(pi_term, nan=0.0, posinf=0.0, neginf=0.0)
            if pi_term.size >= 2:
                top2 = np.partition(pi_term, -2)[-2:]
                terminal_margin = max(float(np.max(top2) - np.min(top2)), 0.0)
            elif pi_term.size == 1:
                terminal_margin = 1.0
            else:
                terminal_margin = 0.0
            total_cost -= xi_weight * terminal_margin_bonus * terminal_margin

        return total_cost

                                                                        
                               
                                                                        

    @staticmethod
    def _sample_scenario_seeds(
        rng: np.random.Generator,
        n_scenarios: int,
    ) -> List[int]:
        return [int(s) for s in rng.integers(0, 2**31, size=n_scenarios)]

    @staticmethod
    def _sample_scenario(
        belief,
        rng: np.random.Generator,
    ) -> Tuple[int, float]:
        n_hyp = len(belief.pi)
        pi = np.asarray(belief.pi, dtype=np.float64)

                                         
        pi = np.clip(pi, 0.0, 1.0)
        pi_sum = pi.sum()
        if pi_sum < 1e-300:
            pi = np.ones(n_hyp, dtype=np.float64) / n_hyp
        else:
            pi = pi / pi_sum

        h = int(rng.choice(n_hyp, p=pi))

        if h == 0:
            c = 0.0
        else:
            mu_h = float(belief.mu[h])
            P_h = float(belief.P[h])
            sigma_h = max(float(np.sqrt(np.abs(P_h))), 1e-15)
            c = float(rng.normal(loc=mu_h, scale=sigma_h))

        return h, c

                                                                        
                                
                                                                        

    def _compute_g_dicts(
        self,
        actions: List[Dict[str, Any]],
    ) -> List[Dict[int, NDArray[np.float64]]]:
        g_dicts: List[Dict[int, NDArray[np.float64]]] = []

        for action in actions:
            cached = action.get("g_dict")
            if isinstance(cached, dict):
                g_dicts.append(cached)
                continue
            H_t = action["H_t"]
            try:
                rs = ResidualSubspace(
                    H_t=H_t,
                    W_sqrt=self.se.W_sqrt,
                    H0=self.attack_model.H0,
                )
                g_dict = rs.compute_all_g(self.attack_model)
                action["g_dict"] = g_dict
            except ValueError:
                                                                      
                                                                       
                logger.debug(
                    "Degenerate residual subspace for action w=%s; "
                    "assigning empty g_dict.",
                    action.get("w", "?"),
                )
                g_dict = {}
                action["g_dict"] = g_dict

            g_dicts.append(g_dict)

        return g_dicts

    def _build_g_norm_matrix(
        self,
        actions: List[Dict[str, Any]],
        g_dicts: List[Dict[int, NDArray[np.float64]]],
    ) -> NDArray[np.float64]:
        n_actions = len(actions)
        n_nonnull = max(int(self.attack_model.n_hypotheses) - 1, 0)
        g_norm_matrix = np.zeros((n_actions, n_nonnull), dtype=np.float64)

        if n_nonnull == 0:
            return g_norm_matrix

        for idx, action in enumerate(actions):
            cached = action.get("_g_norm_sq")
            if isinstance(cached, np.ndarray) and cached.shape == (n_nonnull,):
                g_norm_matrix[idx] = cached
                continue

            g_norm_sq = compute_g_norm_sq_vector(g_dicts[idx], self.attack_model.n_hypotheses)
            action["_g_norm_sq"] = g_norm_sq
            g_norm_matrix[idx] = g_norm_sq

        return g_norm_matrix

    @staticmethod
    def _compute_action_costs(
        actions: List[Dict[str, Any]],
        action_library,
        cfg,
    ) -> NDArray[np.float64]:
        act_costs = np.empty(len(actions), dtype=np.float64)
        for idx, action in enumerate(actions):
            act_costs[idx] = float(
                action.get("cost", action_library.compute_actuation_cost(action, cfg))
            )
        return act_costs

    @staticmethod
    def _compute_detectability_raw_batch_clean(
        belief,
        g_norm_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        g_norm_matrix = np.asarray(g_norm_matrix, dtype=np.float64)
        if g_norm_matrix.size == 0:
            return np.zeros(g_norm_matrix.shape[0], dtype=np.float64)

        detectability_weights = compute_detectability_weights(belief)
        if detectability_weights.size == 0:
            return np.zeros(g_norm_matrix.shape[0], dtype=np.float64)

        with np.errstate(over="ignore", invalid="ignore"):
            d_raw = g_norm_matrix @ detectability_weights
        np.maximum(d_raw, 0.0, out=d_raw)
        return d_raw

    @staticmethod
    def _get_nu(
        g_dict: Dict[int, NDArray[np.float64]],
    ) -> Optional[int]:
        for g_h in g_dict.values():
            return g_h.shape[0]
        return None
