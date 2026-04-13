from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ms_hide.src.sim.episode import EpisodeResult, simulate_episode
from ms_hide.src.utils.metrics import (
    cumulative_actuation_cost,
    detection_success_rate,
    identification_accuracy,
)

logger = logging.getLogger(__name__)

_CANONICAL_METHODS: List[str] = ["MS-HIDE"]
_METHOD_ALIASES: Dict[str, str] = {"DCEE": "MS-HIDE"}


def _normalise_method_name(method_name: str) -> str:
    return _METHOD_ALIASES.get(method_name, method_name)


def _make_planner(
    method_name: str,
    cfg: Any,
    model: Any = None,
    dcse: Any = None,
    attack_model: Any = None,
    action_library: Any = None,
) -> Any:
    if _normalise_method_name(method_name) != "MS-HIDE":
        raise ValueError(f"Unknown method '{method_name}'.")
    if attack_model is None or dcse is None:
        raise ValueError("MS-HIDE requires attack_model and dcse.")
    from ms_hide.src.planner.dcee_planner import DCEEPlanner
    return DCEEPlanner(attack_model, dcse, cfg)


@dataclass
class MethodResults:
    method_name: str = ""
    episodes: List[EpisodeResult] = field(default_factory=list)
    mean_detection_delay: float = 0.0
    detection_rate: float = 0.0
    identification_acc: float = 0.0
    mean_entropy_curve: Any = None
    mean_variance_curve: Any = None
    mean_bdd_statistic_curve: Any = None
    mean_cumulative_cost: float = 0.0
    mean_planner_time: float = 0.0


def _run_single_episode(
    case_cfg: Any,
    planner: Any,
    seed: int,
    model: Any = None,
    dcse: Any = None,
    attack_model: Any = None,
    action_library: Any = None,
) -> EpisodeResult:
    return simulate_episode(
        case_cfg=case_cfg,
        planner=planner,
        seed=seed,
        model=model,
        dcse=dcse,
        attack_model=attack_model,
        action_library=action_library,
    )


_SUITE_CFG: Any = None
_SUITE_MODEL: Any = None
_SUITE_DCSE: Any = None
_SUITE_ATTACK_MODEL: Any = None
_SUITE_ACTION_LIBRARY: Any = None
_SUITE_PLANNER_CACHE: Optional[Dict[str, Any]] = None


def _suite_pool_init(cfg=None, model=None, dcse=None, attack_model=None, action_library=None) -> None:
    global _SUITE_CFG, _SUITE_MODEL, _SUITE_DCSE
    global _SUITE_ATTACK_MODEL, _SUITE_ACTION_LIBRARY, _SUITE_PLANNER_CACHE
    if cfg is not None:
        _SUITE_CFG = cfg
    if model is not None:
        _SUITE_MODEL = model
    if dcse is not None:
        _SUITE_DCSE = dcse
    if attack_model is not None:
        _SUITE_ATTACK_MODEL = attack_model
    if action_library is not None:
        _SUITE_ACTION_LIBRARY = action_library
    _SUITE_PLANNER_CACHE = {}


def _run_single_episode_worker(work_unit):
    method_name, ep_idx, seed = work_unit
    planner_cache = _SUITE_PLANNER_CACHE or {}
    try:
        planner = planner_cache.get(method_name)
        if planner is None:
            planner = _make_planner(
                method_name=method_name,
                cfg=_SUITE_CFG,
                model=_SUITE_MODEL,
                dcse=_SUITE_DCSE,
                attack_model=_SUITE_ATTACK_MODEL,
                action_library=_SUITE_ACTION_LIBRARY,
            )
            planner_cache[method_name] = planner
            globals()["_SUITE_PLANNER_CACHE"] = planner_cache
        ep = _run_single_episode(
            case_cfg=_SUITE_CFG,
            planner=planner,
            seed=seed,
            model=_SUITE_MODEL,
            dcse=_SUITE_DCSE,
            attack_model=_SUITE_ATTACK_MODEL,
            action_library=_SUITE_ACTION_LIBRARY,
        )
        return method_name, ep_idx, True, ep, ""
    except Exception as exc:
        return method_name, ep_idx, False, None, str(exc)


def run_suite(
    case_name: str,
    cfg: Any,
    methods: Optional[List[str]] = None,
    n_episodes: int = 200,
    n_workers: Optional[int] = None,
    model: Any = None,
    dcse: Any = None,
    attack_model: Any = None,
    action_library: Any = None,
    base_seed: int = 42,
) -> Dict[str, MethodResults]:
    del case_name
    requested_methods = list(methods or _CANONICAL_METHODS)
    normalized_methods: List[str] = []
    for name in requested_methods:
        canonical = _normalise_method_name(name)
        if canonical not in _CANONICAL_METHODS:
            logger.warning("Method '%s' is not supported; skipping.", name)
            continue
        if canonical not in normalized_methods:
            normalized_methods.append(canonical)
    if not normalized_methods:
        return {}

    if n_workers is None:
        n_workers = min(10, os.cpu_count() or 1)
    n_workers = max(1, min(int(n_workers), 10, os.cpu_count() or 1))
    seeds = [base_seed + i for i in range(n_episodes)]
    results: Dict[str, MethodResults] = {}

    planners: Dict[str, Any] = {}
    for method_name in normalized_methods:
        try:
            planners[method_name] = _make_planner(
                method_name=method_name,
                cfg=cfg,
                model=model,
                dcse=dcse,
                attack_model=attack_model,
                action_library=action_library,
            )
        except Exception as exc:
            logger.warning("Planner '%s' could not be constructed: %s. Skipping.", method_name, exc)

    active_methods = [m for m in normalized_methods if m in planners]
    if not active_methods:
        return {}

    if n_workers <= 1:
        for method_name in active_methods:
            planner = planners[method_name]
            episodes: List[EpisodeResult] = []
            for seed in seeds:
                episodes.append(
                    _run_single_episode(
                        case_cfg=cfg,
                        planner=planner,
                        seed=seed,
                        model=model,
                        dcse=dcse,
                        attack_model=attack_model,
                        action_library=action_library,
                    )
                )
            results[method_name] = aggregate_method_results(method_name, episodes, cfg.T)
        return results

    import multiprocessing as mp

    global _SUITE_CFG, _SUITE_MODEL, _SUITE_DCSE, _SUITE_ATTACK_MODEL
    global _SUITE_ACTION_LIBRARY, _SUITE_PLANNER_CACHE
    _SUITE_CFG = cfg
    _SUITE_MODEL = model
    _SUITE_DCSE = dcse
    _SUITE_ATTACK_MODEL = attack_model
    _SUITE_ACTION_LIBRARY = action_library
    _SUITE_PLANNER_CACHE = None

    work_units = [
        (method_name, ep_idx, seed)
        for ep_idx, seed in enumerate(seeds)
        for method_name in active_methods
    ]
    episode_map: Dict[str, Dict[int, EpisodeResult]] = {method_name: {} for method_name in active_methods}
    failures: List[tuple] = []
    ctx = mp.get_context("spawn")
    chunksize = max(1, len(work_units) // (max(n_workers, 1) * 8))
    with ctx.Pool(
        processes=n_workers,
        initializer=_suite_pool_init,
        initargs=(cfg, model, dcse, attack_model, action_library),
    ) as pool:
        for method_name, ep_idx, ok, ep, err in pool.imap_unordered(
            _run_single_episode_worker, work_units, chunksize=chunksize
        ):
            if ok:
                episode_map[method_name][ep_idx] = ep
            else:
                failures.append((method_name, ep_idx, err))

    _SUITE_CFG = _SUITE_MODEL = _SUITE_DCSE = None
    _SUITE_ATTACK_MODEL = _SUITE_ACTION_LIBRARY = None
    _SUITE_PLANNER_CACHE = None

    if failures:
        first_method, first_ep, first_err = failures[0]
        raise RuntimeError(
            f"{len(failures)} episodes failed. First failure: {first_method}[episode={first_ep}] -> {first_err}"
        )

    for method_name in active_methods:
        ordered_episodes = [episode_map[method_name][i] for i in range(n_episodes)]
        results[method_name] = aggregate_method_results(method_name, ordered_episodes, cfg.T)
    return results


def run_all_cases(
    cfg_map: Dict[str, Any],
    methods: Optional[List[str]] = None,
    n_episodes: int = 200,
    n_workers: Optional[int] = None,
    prebuilt: Optional[Dict[str, Dict[str, Any]]] = None,
    base_seed: int = 42,
) -> Dict[str, Dict[str, MethodResults]]:
    all_results: Dict[str, Dict[str, MethodResults]] = {}
    for case_name, cfg in cfg_map.items():
        kwargs: Dict[str, Any] = {}
        if prebuilt and case_name in prebuilt:
            kwargs = prebuilt[case_name]
        all_results[case_name] = run_suite(
            case_name=case_name,
            cfg=cfg,
            methods=methods,
            n_episodes=n_episodes,
            n_workers=n_workers,
            base_seed=base_seed,
            **kwargs,
        )
    return all_results


def aggregate_method_results(
    method_name: str,
    episodes: List[EpisodeResult],
    T: int,
) -> MethodResults:
    mr = MethodResults(method_name=method_name, episodes=episodes)
    if not episodes:
        return mr

    delays = [float(ep.detection_delay) for ep in episodes if ep.true_h != 0]
    mr.mean_detection_delay = float(np.mean(delays)) if delays else float(T)

    bdd_flags_list = [ep.bdd_detections for ep in episodes]
    attack_present_list = [ep.true_h != 0 for ep in episodes]
    mr.detection_rate = detection_success_rate(bdd_flags_list, attack_present_list)

    final_beliefs = [ep.belief_history[-1].pi if ep.belief_history else np.zeros(1) for ep in episodes]
    true_hypotheses = [ep.true_h for ep in episodes]
    mr.identification_acc = identification_accuracy(final_beliefs, true_hypotheses)

    entropy_matrix = np.array([ep.entropy_history for ep in episodes])
    variance_matrix = np.array([ep.variance_history for ep in episodes])
    bdd_stat_matrix = np.array([ep.bdd_statistics for ep in episodes])

    mr.mean_entropy_curve = np.mean(entropy_matrix, axis=0)
    mr.mean_variance_curve = np.mean(variance_matrix, axis=0)
    mr.mean_bdd_statistic_curve = np.mean(bdd_stat_matrix, axis=0)

    cum_costs = [cumulative_actuation_cost(ep.actuation_costs) for ep in episodes]
    mr.mean_cumulative_cost = float(np.mean(cum_costs))

    all_times = [t for ep in episodes for t in ep.planner_times]
    mr.mean_planner_time = float(np.mean(all_times)) if all_times else 0.0
    return mr


def aggregate_results(all_results: Dict[str, Dict[str, MethodResults]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"cases": {}, "methods": set()}
    for case_name, method_dict in all_results.items():
        case_summary: Dict[str, Any] = {}
        for method_name, mr in method_dict.items():
            summary["methods"].add(method_name)
            case_summary[method_name] = {
                "mean_detection_delay": mr.mean_detection_delay,
                "detection_rate": mr.detection_rate,
                "identification_accuracy": mr.identification_acc,
                "mean_cumulative_cost": mr.mean_cumulative_cost,
                "mean_planner_time_ms": mr.mean_planner_time * 1000.0,
                "n_episodes": len(mr.episodes),
            }
        summary["cases"][case_name] = case_summary
    summary["methods"] = sorted(summary["methods"])
    return summary
