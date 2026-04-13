
import os
import sys
import warnings
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_REPO_ROOT = os.path.normpath(os.path.join(_PROJECT_ROOT, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.power.matpower_parser import parse_matpower_case
from src.power.dc_model import DCModel
from src.placement.dfs_placement import PlacementSearcher
from src.estimation.belief_update import BeliefUpdater, BeliefState
from src.planner.explore_score import predict_uncertainty
from src.planner.dcee_planner import DCEEPlanner
from ms_hide.src.sim.runner import run_suite
from experiments._common import (
    build_infra,
    make_cfg,
    make_planner,
    run_episode_from_scenario,
)

CASE14_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, os.pardir, "matpower", "data", "case14.m")
)
CASE57_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, os.pardir, "matpower", "data", "case57.m")
)
CASE118_PATH = os.path.normpath(
    os.path.join(_PROJECT_ROOT, os.pardir, "matpower", "data", "case118.m")
)


def _enumerate_simple_cycles_edge_indexed(searcher: PlacementSearcher):
    edges = searcher.edges
    n_bus = searcher.n_buses
    adj = defaultdict(list)
    for e in edges:
        adj[e.u].append((e.v, e.idx))
        adj[e.v].append((e.u, e.idx))

    def canon_cycle(edge_cycle):
        k = len(edge_cycle)
        fwd = tuple(edge_cycle)
        rev = tuple(reversed(edge_cycle))
        rots = [tuple(fwd[i:] + fwd[:i]) for i in range(k)]
        rots += [tuple(rev[i:] + rev[:i]) for i in range(k)]
        return min(rots)

    cycles = set()
    for start in range(n_bus):
        stack = [(start, [start], [], set())]
        while stack:
            node, node_path, edge_path, used_edges = stack.pop()
            for nb, eid in adj[node]:
                if eid in used_edges:
                    continue
                if nb == start and len(edge_path) >= 2:
                    cycles.add(canon_cycle(edge_path + [eid]))
                elif nb not in node_path and len(node_path) < n_bus:
                    stack.append(
                        (nb, node_path + [nb], edge_path + [eid], used_edges | {eid})
                    )
    return [list(c) for c in cycles if len(c) >= 3]


def _max_cyclic_run(flags, target):
    k = len(flags)
    best = 0
    run = 0
    for i in range(2 * k):
        if flags[i % k] == target:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return min(best, k)


def _check_rule_violations_all_cycles(cycles, df_set):
    violations = 0
    for cyc in cycles:
        flags = [eid in df_set for eid in cyc]                      
        if _max_cyclic_run(flags, False) >= 2:
            violations += 1
            continue
        if _max_cyclic_run(flags, True) >= 3:
            violations += 1
    return violations


def test_strict_cycle_mode_detects_infeasibility_case14():
    case_data = parse_matpower_case(CASE14_PATH)
    model = DCModel(case_data)
    searcher = PlacementSearcher(model)

    cycles = _enumerate_simple_cycles_edge_indexed(searcher)
    assert len(cycles) > 0

    relaxed = searcher.find_valid_placement(
        rng=np.random.default_rng(7),
        max_attempts=800,
        strict_all_cycles=False,
    )
    assert relaxed is not None
    assert searcher.check_rules(relaxed, strict_all_cycles=False)

    strict = searcher.find_valid_placement(
        rng=np.random.default_rng(7),
        max_attempts=800,
        strict_all_cycles=True,
    )
    assert strict is None

    n_viol = _check_rule_violations_all_cycles(cycles, relaxed)
    assert n_viol > 0


def test_dc_model_large_cases_no_runtime_warning():
    for case_path in (CASE57_PATH, CASE118_PATH):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            case_data = parse_matpower_case(case_path)
            model = DCModel(case_data)
        assert np.all(np.isfinite(model.z_nom))


def test_predict_uncertainty_changes_with_qc_when_rng_fixed():
    belief = BeliefState(
        pi=np.array([0.2, 0.4, 0.4], dtype=np.float64),
        mu=np.array([0.0, 0.10, -0.10], dtype=np.float64),
        P=np.array([0.0, 0.01, 0.01], dtype=np.float64),
    )
    g_dict = {
        1: np.array([1.0, 0.0], dtype=np.float64),
        2: np.array([0.5, 0.5], dtype=np.float64),
    }

    cfg_low = SimpleNamespace(
        num_scenarios=200,
        lambda_H=0.60,
        prior_c_var=0.01,
        eps_prob=1e-12,
        q_c=1e-9,
    )
    cfg_high = SimpleNamespace(
        num_scenarios=200,
        lambda_H=0.60,
        prior_c_var=0.01,
        eps_prob=1e-12,
        q_c=1.0,
    )
    updater_low = BeliefUpdater(3, cfg_low)
    updater_high = BeliefUpdater(3, cfg_high)

    v_low = predict_uncertainty(
        belief,
        g_dict,
        cfg_low,
        belief_updater=updater_low,
        rng=np.random.default_rng(1234),
    )
    v_high = predict_uncertainty(
        belief,
        g_dict,
        cfg_high,
        belief_updater=updater_high,
        rng=np.random.default_rng(1234),
    )

    assert abs(v_low - v_high) > 1e-6, (
        "predict_uncertainty is unexpectedly insensitive to q_c."
    )


def test_rollout_one_scenario_accepts_full_action_library():
    cfg = SimpleNamespace(
        q_c=1e-5,
        eps_prob=1e-12,
        lambda_H=0.6,
        prior_c_var=0.01,
        rho_u=0.05,
        eps_d=1e-12,
    )
    planner = DCEEPlanner(SimpleNamespace(n_hypotheses=3), None, cfg)

    belief = BeliefState(
        pi=np.array([0.2, 0.4, 0.4], dtype=np.float64),
        mu=np.array([0.0, 0.1, -0.1], dtype=np.float64),
        P=np.array([0.0, 0.01, 0.01], dtype=np.float64),
    )
    actions = [{"id": 0}, {"id": 1}, {"id": 2}]
    all_g_dicts = [
        {1: np.array([1.0, 0.0]), 2: np.array([0.2, 0.1])},
        {1: np.array([0.5, 0.5]), 2: np.array([0.3, 0.2])},
        {1: np.array([0.1, 0.9]), 2: np.array([0.4, 0.4])},
    ]

    class _DummyLib:
        @staticmethod
        def compute_actuation_cost(action, cfg):
            return 0.0

    g_norm_matrix = planner._build_g_norm_matrix(actions, all_g_dicts)
    assert g_norm_matrix.shape[0] == len(actions)

    total_cost = planner._rollout_one_scenario(
        belief=belief,
        action_indices_seq=[0, 1],
        actions=actions,
        all_g_dicts=all_g_dicts,
        h_sample=1,
        c_sample=0.2,
        action_library=_DummyLib(),
        cfg=cfg,
        rng=np.random.default_rng(123),
        g_norm_matrix=g_norm_matrix,
    )
    assert np.isfinite(total_cost)


def test_run_episode_from_scenario_enforces_ground_truth():
    cfg = make_cfg("case14", T=2, max_actions=5, num_random_dirs=6, num_scenarios=3)
    infra = build_infra("case14", cfg)
    planner = make_planner("MS-HIDE", infra, cfg)

    sc_null = {
        "episode_id": 0,
        "attack_flag": False,
        "h_true": 0,
        "c_true": 0.0,
        "noise_seed": 1001,
        "planner_seed": 2001,
    }
    ep0 = run_episode_from_scenario(sc_null, infra, planner, cfg)
    assert ep0.true_h == 0
    assert float(ep0.true_c) == 0.0

    sc_attack = {
        "episode_id": 1,
        "attack_flag": True,
        "h_true": 3,
        "c_true": -0.04,
        "noise_seed": 1002,
        "planner_seed": 2002,
    }
    ep1 = run_episode_from_scenario(sc_attack, infra, planner, cfg)
    assert ep1.true_h == 3
    assert np.isclose(float(ep1.true_c), -0.04)


def test_planner_factory_returns_ms_hide_planner():
    cfg = make_cfg("case14", T=2, max_actions=5, num_random_dirs=6, num_scenarios=3)
    infra = build_infra("case14", cfg)

    planner = make_planner("MS-HIDE", infra, cfg)

    assert type(planner).__module__.startswith("ms_hide.src.planner.dcee_planner")

def test_runner_supports_ms_hide_and_legacy_alias():
    cfg = make_cfg("case14", T=2, max_actions=5, num_random_dirs=6, num_scenarios=3)
    infra = build_infra("case14", cfg)

    requested = [
        "DCEE",
        "MS-HIDE",
    ]
    results = run_suite(
        case_name="case14",
        cfg=cfg,
        methods=requested,
        n_episodes=1,
        n_workers=1,
        model=infra.model,
        dcse=infra.dcse,
        attack_model=infra.attack_model,
        action_library=infra.action_library,
        base_seed=7,
    )

    assert set(results.keys()) == {"MS-HIDE"}
