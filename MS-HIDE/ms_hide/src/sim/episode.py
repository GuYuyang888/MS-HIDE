
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ms_hide.src.power.dc_model import DCModel
from ms_hide.src.power.dc_se import DCSE
from ms_hide.src.attack.fdi_attack import FDIAttackModel
from ms_hide.src.estimation.residual_subspace import ResidualSubspace
from ms_hide.src.estimation.bdd import BDDDetector
from ms_hide.src.estimation.belief_update import (
    BeliefState as BUBeliefState,
    BeliefUpdater,
)
from ms_hide.src.planner.exploit_score import compute_g_norm_sq_vector
from ms_hide.src.utils.log_utils import entropy_normalized
from ms_hide.src.utils.metrics import cumulative_actuation_cost

logger = logging.getLogger(__name__)


                                                                        
                          
                                                                        

@dataclass
class BeliefState:

    pi: NDArray[np.float64]
    mu: NDArray[np.float64]
    P: NDArray[np.float64]

    def copy(self) -> "BeliefState":
        return BeliefState(
            pi=self.pi.copy(),
            mu=self.mu.copy(),
            P=self.P.copy(),
        )


@dataclass
class EpisodeResult:

    bdd_detections: List[bool] = field(default_factory=list)
    bdd_statistics: List[float] = field(default_factory=list)
    belief_history: List[BeliefState] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    variance_history: List[float] = field(default_factory=list)
    actuation_costs: List[float] = field(default_factory=list)
    actions_taken: List[int] = field(default_factory=list)
    true_h: int = 0
    true_c: float = 0.0
    true_h_initial: int = 0
    true_c_initial: float = 0.0
    switch_t: Optional[int] = None
    switch_h: Optional[int] = None
    switch_c: float = 0.0
    final_identification: int = 0
    detection_delay: int = -1
    planner_times: List[float] = field(default_factory=list)
    hide_error_history: List[float] = field(default_factory=list)


                                                                        
                                  
                                                                        

def _initialise_belief(n_hyp: int, cfg: Any) -> BeliefState:
    pi = np.empty(n_hyp, dtype=np.float64)
    pi[0] = cfg.prior_null
    if n_hyp > 1:
        pi[1:] = (1.0 - cfg.prior_null) / (n_hyp - 1)

    mu = np.zeros(n_hyp, dtype=np.float64)
    P = np.zeros(n_hyp, dtype=np.float64)

                                                             
    mu[1:] = cfg.prior_c_mean
    P[1:] = cfg.prior_c_var

    return BeliefState(pi=pi, mu=mu, P=P)


def _predict_belief(belief: BeliefState, cfg: Any) -> BeliefState:
    b = belief.copy()
    q_c = getattr(cfg, "q_c", 1e-5)
    b.P[1:] = b.P[1:] + q_c
    return b


def _update_belief(
    belief: BeliefState,
    y_t: NDArray[np.float64],
    g_dict: Dict[int, NDArray[np.float64]],
    cfg: Any,
    belief_updater: Optional[BeliefUpdater] = None,
) -> BeliefState:
    n_hyp = len(belief.pi)
    updater = belief_updater if belief_updater is not None else BeliefUpdater(n_hyp, cfg)

                                                              
    bu_belief = BUBeliefState(
        pi=belief.pi.copy(),
        mu=belief.mu.copy(),
        P=belief.P.copy(),
    )

                                                         
    bu_updated = updater.update(bu_belief, y_t, g_dict, cfg)

                                         
    return BeliefState(
        pi=bu_updated.pi.copy(),
        mu=bu_updated.mu.copy(),
        P=bu_updated.P.copy(),
    )


                                                                        
                     
                                                                        

def _compute_v_amp(belief: BeliefState, cfg: Any) -> float:
    P_h_0 = float(cfg.prior_c_var)
    if P_h_0 <= 0.0:
        return 0.0
    return float(np.sum(belief.pi[1:] * belief.P[1:] / P_h_0))


                                                                        
                                                  
                                                                        

def _annotate_actions_with_g_dict(
    action_library: Any,
    attack_model: FDIAttackModel,
    dcse: DCSE,
    model: DCModel,
) -> None:
    n_hypotheses = attack_model.n_hypotheses
    for action in action_library.actions:
        if "g_dict" in action:
            if "_g_norm_sq" not in action:
                action["_g_norm_sq"] = compute_g_norm_sq_vector(
                    action["g_dict"],
                    n_hypotheses,
                )
            continue
        H_t = action["H_t"]
        try:
            rs = ResidualSubspace(H_t, dcse.W_sqrt, model.H0)
            g_dict = rs.compute_all_g(attack_model)
            action["g_dict"] = g_dict
            action["_g_norm_sq"] = compute_g_norm_sq_vector(g_dict, n_hypotheses)
            action["_residual_subspace"] = rs
        except ValueError:
                                                                          
            logger.warning(
                "Degenerate residual subspace for action (nu=0). "
                "Marking with empty g_dict."
            )
            action["g_dict"] = {}
            action["_g_norm_sq"] = np.zeros(max(n_hypotheses - 1, 0), dtype=np.float64)
            action["_residual_subspace"] = None


                                                                        
                      
                                                                        

def simulate_episode(
    case_cfg: Any,
    planner: Any,
    seed: int,
    planner_seed: Optional[int] = None,
    measurement_seed: Optional[int] = None,
    attack_seed: Optional[int] = None,
    true_h: Optional[int] = None,
    true_c: Optional[float] = None,
    switch_t: Optional[int] = None,
    switch_h: Optional[int] = None,
    switch_c: Optional[float] = None,
    model: Optional[DCModel] = None,
    dcse: Optional[DCSE] = None,
    attack_model: Optional[FDIAttackModel] = None,
    action_library: Optional[Any] = None,
) -> EpisodeResult:
    rng_master = np.random.default_rng(seed)
    if attack_seed is None:
        attack_seed = int(rng_master.integers(0, 2**31 - 1))
    if planner_seed is None:
        planner_seed = int(rng_master.integers(0, 2**31 - 1))
    if measurement_seed is None:
        measurement_seed = int(rng_master.integers(0, 2**31 - 1))

    rng_attack = np.random.default_rng(int(attack_seed))
    rng_planner = np.random.default_rng(int(planner_seed))
    rng_measurement = np.random.default_rng(int(measurement_seed))
    T: int = case_cfg.T

                                                                        
                                         
                                                                        
    if model is None:
        from ms_hide.src.power.matpower_parser import parse_matpower_case
        case_data = parse_matpower_case(case_cfg.resolve_matpower_path())
        model = DCModel(case_data)

    if dcse is None:
        dcse = DCSE(
            model.z_nom,
            sigma_rel=case_cfg.sigma_rel,
            sigma_min=case_cfg.sigma_min,
        )

    if attack_model is None:
        attack_model = FDIAttackModel(model.H0, model.n)

    if action_library is None:
        raise ValueError(
            "action_library must be provided. Build it externally using "
            "ActionLibrary(model, placement_df_set, U, cfg)."
        )

                                                                        
                                                                      
                                                                        
    if true_h is None:
        true_h, true_c, a_star = attack_model.sample_true_attack(case_cfg, rng_attack)
    else:
        true_h = int(true_h)
        if true_h < 0 or true_h > attack_model.n:
            raise ValueError(
                f"Forced true_h={true_h} is out of range [0, {attack_model.n}]."
            )
        if true_h == 0:
            true_c = 0.0
        else:
            if true_c is None:
                true_c = float(getattr(case_cfg, "true_c", 0.04))
            true_c = float(true_c)
        a_star = attack_model.construct_attack_vector(true_h, true_c)

    true_h_initial = int(true_h)
    true_c_initial = float(true_c)
    switch_attack_enabled = (
        switch_t is not None
        and switch_h is not None
        and 0 <= int(switch_t) < T
    )
    switch_attack_t = int(switch_t) if switch_attack_enabled else None
    switch_attack_h: Optional[int] = None
    switch_attack_c = 0.0
    if switch_attack_enabled:
        switch_attack_h = int(switch_h)
        if switch_attack_h < 0 or switch_attack_h > attack_model.n:
            raise ValueError(
                f"Forced switch_h={switch_attack_h} is out of range [0, {attack_model.n}]."
            )
        if switch_attack_h == 0:
            switch_attack_c = 0.0
        else:
            if switch_c is None:
                switch_c = true_c
            switch_attack_c = float(switch_c)
        a_switch = attack_model.construct_attack_vector(switch_attack_h, switch_attack_c)
        true_h = switch_attack_h
        true_c = switch_attack_c
    else:
        a_switch = a_star
    attack_present = (true_h_initial != 0) or (switch_attack_h is not None and switch_attack_h != 0)

    logger.debug(
        "Episode seed=%d: true_h=%d, true_c=%.6f, attack_present=%s",
        seed, true_h, true_c, attack_present,
    )

                                                                        
                                                       
                                                                        
    if not bool(getattr(action_library, "_g_precomputed", False)):
        _annotate_actions_with_g_dict(action_library, attack_model, dcse, model)
        action_library._g_precomputed = True

    action_index_by_id = {
        id(action): idx for idx, action in enumerate(action_library.actions)
    }

                                                                        
                          
                                                                        
    n_hyp = attack_model.n_hypotheses
    belief = _initialise_belief(n_hyp, case_cfg)
    belief_updater = BeliefUpdater(n_hyp, case_cfg)

                                                                        
                                                                    
                                                                        
    zero_action = action_library.zero_action
    zero_rs = zero_action.get("_residual_subspace")
    if zero_rs is None:
        zero_rs = ResidualSubspace(zero_action["H_t"], dcse.W_sqrt, model.H0)
    bdd = BDDDetector(nu=zero_rs.nu, alpha_fa=case_cfg.alpha_fa)

                                                                        
                     
                                                                        
    result = EpisodeResult(
        true_h=int(true_h),
        true_c=float(true_c),
        true_h_initial=true_h_initial,
        true_c_initial=true_c_initial,
        switch_t=switch_attack_t,
        switch_h=switch_attack_h,
        switch_c=float(switch_attack_c),
    )
    oracle_mode: bool = getattr(case_cfg, "oracle_hiddenness", True)
    first_detection: Optional[int] = None

    for t in range(T):
                                     
        t_plan_start = time.perf_counter()

        if t == 0:
                                                                           
            action = action_library.zero_action
            action_idx = 0
        else:
            action, _ = planner.plan(belief, action_library, case_cfg, rng_planner)
                                                                                    
            action_idx = action_index_by_id.get(id(action))
            if action_idx is None:
                action_idx = 0
                for j, a in enumerate(action_library.actions):
                    if a is action:
                        action_idx = j
                        break

        t_plan_end = time.perf_counter()
        planner_time = t_plan_end - t_plan_start

                                                                  
        H_t = action["H_t"]
        rs = action.get("_residual_subspace")
        if rs is None:
            try:
                rs = ResidualSubspace(H_t, dcse.W_sqrt, model.H0)
            except ValueError:
                                                       
                logger.warning(
                    "Degenerate residual subspace at t=%d; "
                    "falling back to zero action.", t,
                )
                action = action_library.zero_action
                H_t = action["H_t"]
                rs = action.get("_residual_subspace")
                if rs is None:
                    rs = ResidualSubspace(H_t, dcse.W_sqrt, model.H0)
                action_idx = 0

        g_dict = action.get("g_dict", {})

                                                                                
        if rs.nu != bdd.nu:
            bdd = BDDDetector(nu=rs.nu, alpha_fa=case_cfg.alpha_fa)

                                                
        active_attack = a_star
        if switch_attack_enabled and switch_attack_t is not None and t >= switch_attack_t:
            active_attack = a_switch

        if oracle_mode:
                                                          
                                                                             
                                                            
            e_t = rng_measurement.normal(loc=0.0, scale=dcse.sigma)
            z_t = model.z_nom + active_attack + e_t
            hide_error = 0.0
        else:
                                                                       
                                                                         
            theta_actual = action.get("theta_shifted", model.theta0)
            honest_nominal = H_t @ theta_actual
            delta_z_hide = honest_nominal - model.z_nom
            if np.all(np.isfinite(delta_z_hide)):
                hide_error = float(np.max(np.abs(delta_z_hide)))
            else:
                hide_error = float("inf")
            if hide_error > float(getattr(case_cfg, "hide_tol", 1e-8)):
                logger.debug(
                    "Non-oracle hiddenness mismatch at t=%d: ||delta_z||_inf=%.3e",
                    t,
                    hide_error,
                )
            z_t = dcse.generate_measurement(
                H_t, theta_actual,
                attack_vector=active_attack,
                rng=rng_measurement,
            )

                                                    
        y_t = rs.project(z_t)

                                
        xi_t = bdd.compute_statistic(y_t)
        detected = bdd.detect(y_t)

        if detected and first_detection is None:
            first_detection = t

                                                          
        belief = _predict_belief(belief, case_cfg)
        belief = _update_belief(
            belief,
            y_t,
            g_dict,
            case_cfg,
            belief_updater=belief_updater,
        )

                                       
        h_supp = entropy_normalized(belief.pi, n_hyp, eps_pi=getattr(case_cfg, "eps_prob", 1e-12))
        v_amp = _compute_v_amp(belief, case_cfg)
        act_cost = float(
            action.get("cost", action_library.compute_actuation_cost(action, case_cfg))
        )

                              
        result.bdd_detections.append(detected)
        result.bdd_statistics.append(xi_t)
        result.belief_history.append(belief.copy())
        result.entropy_history.append(h_supp)
        result.variance_history.append(v_amp)
        result.actuation_costs.append(act_cost)
        result.actions_taken.append(action_idx)
        result.planner_times.append(planner_time)
        result.hide_error_history.append(hide_error)

                                                                        
                         
                                                                        
    result.final_identification = int(np.argmax(belief.pi))
    if first_detection is not None:
        result.detection_delay = first_detection
    else:
        result.detection_delay = T                   

    logger.debug(
        "Episode seed=%d completed: detection_delay=%d, "
        "final_id=%d (true=%d), cum_cost=%.6f",
        seed,
        result.detection_delay,
        result.final_identification,
        true_h,
        cumulative_actuation_cost(result.actuation_costs),
    )

    return result
