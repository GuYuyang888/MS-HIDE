from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ms_hide.experiments._common import build_infra, make_cfg, results_dir, run_algorithms_parallel
from ms_hide.experiments.scenario_bank import generate_attack_only_bank
from ms_hide.src.placement.dfs_placement import PlacementSearcher

logger = logging.getLogger(__name__)

CASES = ["case14", "case57", "case118"]
RUNTIME_ALGOS = ["MS-HIDE"]
N_EPISODES_RUNTIME = 20
SEED = 42
RUNTIME_CASE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "case57": {"num_random_dirs": 220, "max_actions": 40},
    "case118": {"num_scenarios": 12, "num_random_dirs": 220, "max_actions": 40},
}


def run(cfg_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if cfg_overrides is None:
        cfg_overrides = {}

    out_dir = results_dir("exp7_scalability")
    rows: List[Dict[str, Any]] = []

    for case_name in CASES:
        logger.info("=" * 60)
        logger.info("Exp7: %s", case_name)
        logger.info("=" * 60)
        case_overrides = dict(RUNTIME_CASE_OVERRIDES.get(case_name, {}))
        case_overrides.update(cfg_overrides)
        cfg = make_cfg(case_name, **case_overrides)
        infra = build_infra(case_name, cfg)

        max_hide_error = 0.0
        n_valid = 0
        for action in infra.action_library.actions:
            if action.get("hidden", True):
                n_valid += 1
            hide_err = action.get("hide_error", 0.0)
            if isinstance(hide_err, (int, float)):
                max_hide_error = max(max_hide_error, abs(hide_err))
            theta_shifted = action.get("theta_shifted")
            if theta_shifted is not None:
                theta_shifted = np.asarray(theta_shifted, dtype=float)
                if np.all(np.isfinite(theta_shifted)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        honest = action.get("H_t", infra.model.H0) @ theta_shifted
                        delta = honest - infra.model.z_nom
                    finite_delta = np.asarray(delta, dtype=float)
                    if np.any(np.isfinite(finite_delta)):
                        max_hide_error = max(
                            max_hide_error,
                            float(np.nanmax(np.abs(finite_delta))),
                        )

        valid_ratio = n_valid / len(infra.action_library) if len(infra.action_library) > 0 else 0.0
        strict_rule_ok = PlacementSearcher(infra.model).check_rules(infra.placement_df_set, strict_all_cycles=True)
        bank = generate_attack_only_bank(case_name, n_attack=N_EPISODES_RUNTIME, c0=cfg.true_c, seed=SEED)
        runtime_rows = run_algorithms_parallel(RUNTIME_ALGOS, infra, bank, cfg, episode_timeout=0, log=logger)
        runtime_df = pd.DataFrame(runtime_rows)
        ms_plan_time = float(runtime_df["planner_runtime_ms_mean"].mean()) if len(runtime_df) else 0.0

        rows.append(
            {
                "case": case_name,
                "n_bus": infra.model.n_bus,
                "n_l": infra.model.n_l,
                "n_DF": len(infra.placement_df_set),
                "free_dim_s": infra.s,
                "n_actions": len(infra.action_library),
                "placement_time_s": infra.placement_time,
                "library_time_s": infra.library_time,
                "max_hide_error": max_hide_error,
                "valid_action_ratio": valid_ratio,
                "strict_rule_ok": bool(strict_rule_ok),
                "strict_cycle_mode": bool(getattr(cfg, "strict_cycle_rules", False)),
                "MS-HIDE_plan_time_ms": ms_plan_time,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table2_scalability_summary.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "scalability_summary.csv"), index=False)
    with open(os.path.join(out_dir, "table2_scalability_latex.txt"), "w", encoding="utf-8") as handle:
        handle.write(df.to_latex(index=False, float_format=lambda value: f"{value:.4f}"))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
