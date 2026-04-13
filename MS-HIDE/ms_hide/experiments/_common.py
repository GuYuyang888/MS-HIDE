
from __future__ import annotations

import copy
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

                                                                             
                                      
                                                                             
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
                                                                          
                     
_OUTPUT_ROOT = _REPO_ROOT
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ms_hide.config.cfg_global import GlobalConfig
from ms_hide.config.cfg_case14 import make_config as make_cfg14
from ms_hide.config.cfg_case57 import make_config as make_cfg57
from ms_hide.config.cfg_case118 import make_config as make_cfg118
from ms_hide.src.power.matpower_parser import parse_matpower_case
from ms_hide.src.power.dc_model import DCModel
from ms_hide.src.power.dc_se import DCSE
from ms_hide.src.attack.fdi_attack import FDIAttackModel
from ms_hide.src.placement.dfs_placement import find_hidden_placements, PlacementSearcher
from ms_hide.src.hmtd.component_basis import build_component_basis_U
from ms_hide.src.hmtd.action_library import ActionLibrary
from ms_hide.src.sim.episode import (
    simulate_episode,
    EpisodeResult,
    _annotate_actions_with_g_dict,
)
from ms_hide.src.utils.metrics import cumulative_actuation_cost

from ms_hide.experiments.scenario_bank import resolve_h_true

logger = logging.getLogger(__name__)


                                                                        
                        
                                                                        

ALGO_COLORS = {"MS-HIDE": "#D62728"}

ALGO_LINESTYLES = {"MS-HIDE": "-"}

ALGO_MARKERS = {"MS-HIDE": "o"}

BOOTSTRAP_RESAMPLES = 2000


                                                                        
                
                                                                        

CASE_CONFIG_MAP = {
    "case14": make_cfg14,
    "case57": make_cfg57,
    "case118": make_cfg118,
}


def make_cfg(case_name: str, **overrides) -> GlobalConfig:
    maker = CASE_CONFIG_MAP[case_name]
    return maker(**overrides)


_EXP_DIR_MAP = {
    "exp1_overall":              "01-总体性能对比",
    "exp2_attack_strength":      "02-攻击强度敏感性",
    "exp3_target_bus":           "03-逐目标母线分析",
    "exp5_budget_tradeoff":      "04-预算权衡分析",
    "exp6_noise_robustness":     "05-噪声鲁棒性",
    "exp7_scalability":          "06-可扩展性验证",
}


def results_dir(exp_name: str) -> str:
    folder = _EXP_DIR_MAP.get(exp_name, exp_name)
    d = os.path.join(_OUTPUT_ROOT, "results", folder)
    os.makedirs(d, exist_ok=True)
    return d


@dataclass
class CaseInfra:
    case_name: str
    cfg: GlobalConfig
    model: DCModel
    dcse: DCSE
    attack_model: FDIAttackModel
    placement_df_set: Set[int]
    U: np.ndarray
    s: int
    action_library: ActionLibrary
    placement_time: float = 0.0
    library_time: float = 0.0


def build_infra(case_name: str, cfg: Optional[GlobalConfig] = None,
                **cfg_overrides) -> CaseInfra:
    if cfg is None:
        cfg = make_cfg(case_name, **cfg_overrides)

                                          
    matpower_path = cfg.resolve_matpower_path(_PROJECT_ROOT)
    if not os.path.exists(matpower_path):
        raise FileNotFoundError(
            f"{case_name}: MATPOWER case file does not exist: {matpower_path}"
        )
    case_data = parse_matpower_case(matpower_path)
    model = DCModel(case_data)
    logger.info("%s: loaded MATPOWER case from %s", case_name, matpower_path)

                        
    dcse = DCSE(model.z_nom, sigma_rel=cfg.sigma_rel, sigma_min=cfg.sigma_min)

                     
    attack_model = FDIAttackModel(model.H0, model.n)

                          
    t0 = time.perf_counter()
    placements = find_hidden_placements(model, cfg, n_placements=1)
    placement_time = time.perf_counter() - t0
    if not placements:
        raise RuntimeError(f"No valid placement found for {case_name}.")
    placement_df_set = placements[0]
    strict_mode = bool(getattr(cfg, "strict_cycle_rules", False))
    strict_ok = PlacementSearcher(model).check_rules(
        placement_df_set, strict_all_cycles=True
    )
    if strict_mode and not strict_ok:
        raise RuntimeError(
            f"{case_name}: strict_cycle_rules=True but selected placement "
            "does not satisfy strict all-cycle checks."
        )
    if (not strict_mode) and (not strict_ok):
        logger.warning(
            "%s: placement satisfies fundamental-cycle checks but not strict all-cycle checks.",
            case_name,
        )

                        
    zero_flow_tol = float(getattr(cfg, "zero_flow_tol", 0.01))
    U, components, s = build_component_basis_U(
        placement_df_set, model, zero_flow_tol=zero_flow_tol
    )

                       
    t0 = time.perf_counter()
    action_library = ActionLibrary(model, placement_df_set, U, cfg)
    _annotate_actions_with_g_dict(action_library, attack_model, dcse, model)
    action_library._g_precomputed = True
    library_time = time.perf_counter() - t0

    logger.info(
        "%s: n=%d, m=%d, n_l=%d, |DF|=%d, s=%d, |A|=%d, "
        "placement=%.3fs, library=%.3fs",
        case_name, model.n, model.m, model.n_l,
        len(placement_df_set), s, len(action_library),
        placement_time, library_time,
    )

    return CaseInfra(
        case_name=case_name, cfg=cfg,
        model=model, dcse=dcse, attack_model=attack_model,
        placement_df_set=placement_df_set, U=U, s=s,
        action_library=action_library,
        placement_time=placement_time,
        library_time=library_time,
    )


                                                                        
                 
                                                                        

def make_planner(algo_name: str, infra: CaseInfra, cfg: Optional[GlobalConfig] = None):
    if cfg is None:
        cfg = infra.cfg

    canonical = "MS-HIDE" if algo_name == "DCEE" else algo_name
    if canonical != "MS-HIDE":
        raise ValueError(f"Unknown algorithm: {algo_name}")
    from ms_hide.src.planner.dcee_planner import DCEEPlanner
    return DCEEPlanner(infra.attack_model, infra.dcse, cfg)


                                                                        
                                           
                                                                        

def run_episode_from_scenario(
    scenario: Dict[str, Any],
    infra: CaseInfra,
    planner: Any,
    cfg: Optional[GlobalConfig] = None,
) -> EpisodeResult:
    if cfg is None:
        cfg = infra.cfg

    noise_seed = int(scenario["noise_seed"])
    planner_seed = int(scenario.get("planner_seed", noise_seed + 1))

                                                                                 
    attack_flag = bool(scenario.get("attack_flag", True))
    switch_t = scenario.get("switch_t", None)
    switch_h = None
    switch_c = None
    if not attack_flag:
        true_h = 0
        true_c = 0.0
    else:
        h_rng = np.random.default_rng(int(scenario.get("h_seed", noise_seed)))
        true_h = int(resolve_h_true(scenario, infra.model.n, h_rng))
        true_c = float(scenario.get("c_true", getattr(cfg, "true_c", 0.04)))
        if true_h == 0:
            true_c = 0.0
        if switch_t is not None:
            raw_switch_h = int(scenario.get("switch_h_true", -1))
            if raw_switch_h == -1:
                switch_seed = int(scenario.get("switch_h_seed", noise_seed + 17))
                switch_rng = np.random.default_rng(switch_seed)
                switch_scenario = {"h_true": -1, "h_seed": switch_seed}
                switch_h = int(resolve_h_true(switch_scenario, infra.model.n, switch_rng))
            else:
                switch_h = raw_switch_h
            if switch_h == true_h and infra.model.n > 1:
                                                        
                switch_h = 1 + (int(true_h) % int(infra.model.n))
            switch_c = float(scenario.get("switch_c_true", true_c))

    return simulate_episode(
        case_cfg=cfg,
        planner=planner,
        seed=noise_seed,
        planner_seed=planner_seed,
        true_h=true_h,
        true_c=true_c,
        switch_t=switch_t,
        switch_h=switch_h,
        switch_c=switch_c,
        model=infra.model,
        dcse=infra.dcse,
        attack_model=infra.attack_model,
        action_library=infra.action_library,
    )


                                                                        
                                          
                                                                        

def extract_episode_metrics(
    ep: EpisodeResult,
    cfg: GlobalConfig,
) -> Dict[str, Any]:
    T = len(ep.bdd_detections)
    attack_present = ep.true_h != 0
    detected = any(ep.bdd_detections)
    false_alarm = detected and not attack_present

    return {
        "attack_flag": attack_present,
        "h_true": ep.true_h,
        "c_true": ep.true_c,
        "detected": detected,
        "tau_detect": ep.detection_delay,
        "tau_detect_censored": min(ep.detection_delay, T + 1),
        "false_alarm": false_alarm,
        "h_hat_final": ep.final_identification,
        "id_correct": ep.final_identification == ep.true_h,
        "final_entropy": ep.entropy_history[-1] if ep.entropy_history else 1.0,
        "final_ampvar": ep.variance_history[-1] if ep.variance_history else 1.0,
        "cumulative_act_cost": cumulative_actuation_cost(ep.actuation_costs),
        "planner_runtime_ms_mean": float(np.mean(ep.planner_times)) * 1000.0 if ep.planner_times else 0.0,
        "planner_runtime_ms_p95": float(np.percentile(ep.planner_times, 95)) * 1000.0 if ep.planner_times else 0.0,
        "max_hide_error": float(np.max(ep.hide_error_history)) if ep.hide_error_history else 0.0,
    }


                                                                        
              
                                                                        

def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    ci: float = 0.95,
    seed: int = 999,
) -> Tuple[float, float, float]:
    data = np.asarray(data, dtype=np.float64)
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    point = float(stat_fn(data))
    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = float(stat_fn(sample))
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_stats, 100 * alpha))
    ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    return point, ci_low, ci_high


                                                                        
                                          
                                                                        

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })
    return plt


                                                                        
                                        
                                                                        

CPU_ONLY_ALGORITHMS = frozenset({"MS-HIDE"})

                                                    
                                                                            
                                                        
_PAR_INFRA: Optional[CaseInfra] = None
_PAR_BANK: Optional[list] = None
_PAR_CFG: Optional[GlobalConfig] = None
_PAR_TIMEOUT: int = 0
_PAR_COLLECT_TRACES: bool = False
_PAR_TRACE_T: int = 0
_PAR_PLANNER_CACHE: Optional[Dict[str, Any]] = None
                                                                         
_PARALLEL_WORKERS: int = min(10, os.cpu_count() or 1)


def _par_worker_init(infra=None, bank=None, cfg=None, timeout=0,
                     collect_traces=False, trace_t=0) -> None:
    global _PAR_INFRA, _PAR_BANK, _PAR_CFG, _PAR_TIMEOUT
    global _PAR_COLLECT_TRACES, _PAR_TRACE_T, _PAR_PLANNER_CACHE
    if infra is not None:
        _PAR_INFRA = infra
    if bank is not None:
        _PAR_BANK = bank
    if cfg is not None:
        _PAR_CFG = cfg
    _PAR_TIMEOUT = int(timeout)
    _PAR_COLLECT_TRACES = bool(collect_traces)
    _PAR_TRACE_T = int(trace_t)
    _PAR_PLANNER_CACHE = {}


def _par_worker(work_unit):
    import signal

    algo_name, episode_idx = work_unit

    infra = _PAR_INFRA
    bank = _PAR_BANK
    cfg = _PAR_CFG
    timeout = _PAR_TIMEOUT
    collect_traces = _PAR_COLLECT_TRACES
    trace_t = _PAR_TRACE_T

    planner_cache = _PAR_PLANNER_CACHE
    if planner_cache is None:
        planner_cache = {}

    class _Timeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _Timeout

    old_handler = None
    try:
        planner = planner_cache.get(algo_name)
        if planner is None:
            planner = make_planner(algo_name, infra, cfg)
            planner_cache[algo_name] = planner
                                                                    
            globals()["_PAR_PLANNER_CACHE"] = planner_cache

        scenario = bank[episode_idx]
        episode_id = int(scenario.get("episode_id", episode_idx))
        if timeout > 0:
            old_handler = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(timeout)
        ep = run_episode_from_scenario(scenario, infra, planner, cfg)
        if timeout > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        metrics = extract_episode_metrics(ep, cfg)
        metrics["algorithm"] = algo_name
        metrics["episode_id"] = episode_id

        if collect_traces:
            t_cap = trace_t if trace_t > 0 else len(ep.entropy_history)
            entropy_trace = list(ep.entropy_history[:t_cap])
            detect_trace = list(ep.bdd_detections[:t_cap])
        else:
            entropy_trace = None
            detect_trace = None

        return algo_name, episode_id, True, metrics, entropy_trace, detect_trace

    except _Timeout:
        if timeout > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        return algo_name, int(episode_idx), False, "timeout", None, None
    except Exception as exc:
        if timeout > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        return algo_name, int(episode_idx), False, str(exc), None, None


def _run_algorithms_parallel_internal(
    algorithms: List[str],
    infra: CaseInfra,
    bank: list,
    cfg: GlobalConfig,
    episode_timeout: int = 0,
    log: Optional[logging.Logger] = None,
    collect_traces: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[List[float]]], Dict[str, List[List[bool]]]]:
    import multiprocessing as mp
    import signal
    from collections import defaultdict

    if log is None:
        log = logger

    cpu_algos = [a for a in algorithms if a in CPU_ONLY_ALGORITHMS]
    gpu_algos = [a for a in algorithms if a not in CPU_ONLY_ALGORITHMS]

                                                                    
    global _PAR_INFRA, _PAR_BANK, _PAR_CFG, _PAR_TIMEOUT
    global _PAR_COLLECT_TRACES, _PAR_TRACE_T, _PAR_PLANNER_CACHE
    _PAR_INFRA = infra
    _PAR_BANK = bank
    _PAR_CFG = cfg
    _PAR_TIMEOUT = int(episode_timeout)
    _PAR_COLLECT_TRACES = bool(collect_traces)
    _PAR_TRACE_T = int(getattr(cfg, "T", 0))
    _PAR_PLANNER_CACHE = None

    all_rows: List[Dict[str, Any]] = []
    entropy_pairs: Dict[str, List[Tuple[int, List[float]]]] = defaultdict(list)
    detect_pairs: Dict[str, List[Tuple[int, List[bool]]]] = defaultdict(list)
    algo_ok: Dict[str, int] = defaultdict(int)
    algo_fail: Dict[str, int] = defaultdict(int)

    n_workers = max(1, min(_PARALLEL_WORKERS, mp.cpu_count() or 1))
                                                                               
    if os.environ.get("MSHIDE_FORCE_SERIAL_POOL", "0") == "1":
        n_workers = 1
    n_episodes = len(bank)

                                                                           
                                                                      
    work_units: List[Tuple[str, int]] = [
        (algo_name, ep_idx)
        for ep_idx in range(n_episodes)
        for algo_name in cpu_algos
    ]

    if work_units:
        if n_workers <= 1:
            log.info(
                "  Serial fallback: %d work units on 1 worker (algos: %s)",
                len(work_units),
                cpu_algos,
            )
            _par_worker_init()
            for (
                algo_name,
                episode_id,
                ok,
                payload,
                entropy_trace,
                detect_trace,
            ) in map(_par_worker, work_units):
                if ok:
                    all_rows.append(payload)
                    algo_ok[algo_name] += 1
                    if collect_traces:
                        entropy_pairs[algo_name].append((episode_id, entropy_trace))
                        detect_pairs[algo_name].append((episode_id, detect_trace))
                else:
                    algo_fail[algo_name] += 1
        else:
                                                                  
                                                                      
            ctx = mp.get_context("spawn")
            chunksize = max(1, len(work_units) // (n_workers * 8))
            log.info(
                "  Parallel pool: %d work units across %d workers (flattened alg×ep queue; algos: %s)",
                len(work_units),
                n_workers,
                cpu_algos,
            )
            with ctx.Pool(
                processes=n_workers,
                initializer=_par_worker_init,
                initargs=(_PAR_INFRA, _PAR_BANK, _PAR_CFG,
                          _PAR_TIMEOUT, _PAR_COLLECT_TRACES, _PAR_TRACE_T),
            ) as pool:
                for (
                    algo_name,
                    episode_id,
                    ok,
                    payload,
                    entropy_trace,
                    detect_trace,
                ) in pool.imap_unordered(_par_worker, work_units, chunksize=chunksize):
                    if ok:
                        all_rows.append(payload)
                        algo_ok[algo_name] += 1
                        if collect_traces:
                            entropy_pairs[algo_name].append((episode_id, entropy_trace))
                            detect_pairs[algo_name].append((episode_id, detect_trace))
                    else:
                        algo_fail[algo_name] += 1

                                                                      
    for algo in gpu_algos:
        log.info("  GPU algo: %s (%d episodes)...", algo, len(bank))
        planner = make_planner(algo, infra, cfg)
        t0 = time.perf_counter()
        for idx, scenario in enumerate(bank):
            episode_id = int(scenario.get("episode_id", idx))
            try:
                if episode_timeout > 0:
                    old_h = signal.signal(signal.SIGALRM, _alarm_handler)
                    signal.alarm(episode_timeout)
                ep = run_episode_from_scenario(scenario, infra, planner, cfg)
                if episode_timeout > 0:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_h)
                metrics = extract_episode_metrics(ep, cfg)
                metrics["algorithm"] = algo
                metrics["episode_id"] = episode_id
                all_rows.append(metrics)
                algo_ok[algo] += 1
                if collect_traces:
                    t_cap = int(getattr(cfg, "T", 0))
                    entropy_pairs[algo].append((episode_id, list(ep.entropy_history[:t_cap])))
                    detect_pairs[algo].append((episode_id, list(ep.bdd_detections[:t_cap])))
            except Exception:
                algo_fail[algo] += 1
            if (idx + 1) % 50 == 0 or (idx + 1) == len(bank):
                elapsed = time.perf_counter() - t0
                rate = (idx + 1) / max(elapsed, 1e-9)
                log.info(
                    "    %s: %d/%d (%.1f ep/s)",
                    algo, idx + 1, len(bank), rate,
                )

    for algo in algorithms:
        log.info("  %s: %d OK, %d failed", algo, algo_ok[algo], algo_fail[algo])

                                                             
    all_rows.sort(
        key=lambda r: (
            str(r.get("algorithm", "")),
            int(r.get("episode_id", -1)),
        )
    )

    entropy_traces: Dict[str, List[List[float]]] = {}
    detect_traces: Dict[str, List[List[bool]]] = {}
    if collect_traces:
        for algo in algorithms:
            ent_sorted = sorted(entropy_pairs.get(algo, []), key=lambda x: x[0])
            det_sorted = sorted(detect_pairs.get(algo, []), key=lambda x: x[0])
            entropy_traces[algo] = [trace for _, trace in ent_sorted]
            detect_traces[algo] = [trace for _, trace in det_sorted]

                          
    _PAR_INFRA = _PAR_BANK = _PAR_CFG = None
    _PAR_TIMEOUT = 0
    _PAR_COLLECT_TRACES = False
    _PAR_TRACE_T = 0
    _PAR_PLANNER_CACHE = None

    return all_rows, entropy_traces, detect_traces


def run_algorithms_parallel(
    algorithms: List[str],
    infra: CaseInfra,
    bank: list,
    cfg: GlobalConfig,
    episode_timeout: int = 0,
    log: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    rows, _, _ = _run_algorithms_parallel_internal(
        algorithms=algorithms,
        infra=infra,
        bank=bank,
        cfg=cfg,
        episode_timeout=episode_timeout,
        log=log,
        collect_traces=False,
    )
    return rows


def run_algorithms_parallel_with_traces(
    algorithms: List[str],
    infra: CaseInfra,
    bank: list,
    cfg: GlobalConfig,
    episode_timeout: int = 0,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[List[float]]], Dict[str, List[List[bool]]]]:
    return _run_algorithms_parallel_internal(
        algorithms=algorithms,
        infra=infra,
        bank=bank,
        cfg=cfg,
        episode_timeout=episode_timeout,
        log=log,
        collect_traces=True,
    )


class _AlarmTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _AlarmTimeout
