from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ms_hide.experiments._common import build_infra, bootstrap_ci, make_cfg, results_dir, run_algorithms_parallel
from ms_hide.experiments.scenario_bank import generate_scenario_bank

logger = logging.getLogger(__name__)

CASES = ["case14", "case57", "case118"]
ALGORITHMS = ["MS-HIDE"]
N_ATTACK = 200
N_NULL = 50
SEED = 42


def run(cfg_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if cfg_overrides is None:
        cfg_overrides = {}

    out_dir = results_dir("exp1_overall")
    all_rows: List[Dict[str, Any]] = []

    for case_name in CASES:
        logger.info("=" * 60)
        logger.info("Exp1: %s", case_name)
        logger.info("=" * 60)
        cfg = make_cfg(case_name, **cfg_overrides)
        infra = build_infra(case_name, cfg)
        bank = generate_scenario_bank(case_name, n_attack=N_ATTACK, n_null=N_NULL, cfg=cfg, seed=SEED)
        timeout = 180 if case_name == "case118" else 0
        rows = run_algorithms_parallel(ALGORITHMS, infra, bank, cfg, episode_timeout=timeout, log=logger)
        for row in rows:
            row["case"] = case_name
        all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(os.path.join(out_dir, "raw_episode_metrics.csv"), index=False)

    summary_rows = []
    for case_name in CASES:
        sub = df_raw[df_raw["case"] == case_name]
        if len(sub) == 0:
            continue
        atk = sub[sub["attack_flag"] == True]
        null = sub[sub["attack_flag"] == False]
        sr_d, sr_d_lo, sr_d_hi = bootstrap_ci(atk["detected"].values.astype(float)) if len(atk) else (0.0, 0.0, 0.0)
        delay, delay_lo, delay_hi = bootstrap_ci(atk["tau_detect_censored"].values.astype(float)) if len(atk) else (0.0, 0.0, 0.0)
        far, far_lo, far_hi = bootstrap_ci(null["false_alarm"].values.astype(float)) if len(null) else (0.0, 0.0, 0.0)
        acc_id, acc_lo, acc_hi = bootstrap_ci(atk["id_correct"].values.astype(float)) if len(atk) else (0.0, 0.0, 0.0)
        ent, ent_lo, ent_hi = bootstrap_ci(atk["final_entropy"].values) if len(atk) else (0.0, 0.0, 0.0)
        cost, cost_lo, cost_hi = bootstrap_ci(atk["cumulative_act_cost"].values) if len(atk) else (0.0, 0.0, 0.0)
        plan_t = float(atk["planner_runtime_ms_mean"].mean()) if len(atk) else 0.0
        summary_rows.append(
            {
                "case": case_name,
                "algorithm": "MS-HIDE",
                "SR_D": sr_d,
                "SR_D_ci_low": sr_d_lo,
                "SR_D_ci_high": sr_d_hi,
                "delay_cens_mean": delay,
                "delay_ci_low": delay_lo,
                "delay_ci_high": delay_hi,
                "FAR": far,
                "FAR_ci_low": far_lo,
                "FAR_ci_high": far_hi,
                "Acc_ID": acc_id,
                "Acc_ID_ci_low": acc_lo,
                "Acc_ID_ci_high": acc_hi,
                "final_entropy_mean": ent,
                "entropy_ci_low": ent_lo,
                "entropy_ci_high": ent_hi,
                "act_cost_mean": cost,
                "cost_ci_low": cost_lo,
                "cost_ci_high": cost_hi,
                "planner_time_ms": plan_t,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "table1_overall_summary.csv"), index=False)
    with open(os.path.join(out_dir, "table1_overall_latex.txt"), "w", encoding="utf-8") as handle:
        handle.write(df_summary.to_latex(index=False, float_format=lambda value: f"{value:.4f}"))
    return df_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
