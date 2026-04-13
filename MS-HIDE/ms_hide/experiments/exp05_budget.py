from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ms_hide.experiments._common import ALGO_COLORS, ALGO_LINESTYLES, ALGO_MARKERS, build_infra, bootstrap_ci, make_cfg, results_dir, run_algorithms_parallel, setup_matplotlib
from ms_hide.experiments.scenario_bank import generate_attack_only_bank

logger = logging.getLogger(__name__)

CASE = "case57"
ETA_X_VALUES = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.20]
ALGORITHMS = ["MS-HIDE"]
N_ATTACK = 200
SEED = 42


def run(cfg_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if cfg_overrides is None:
        cfg_overrides = {}

    out_dir = results_dir("exp5_budget_tradeoff")
    all_rows: List[Dict[str, Any]] = []
    for eta_x in ETA_X_VALUES:
        logger.info("=" * 60)
        logger.info("Exp5: eta_x = %.3f", eta_x)
        logger.info("=" * 60)
        cfg = make_cfg(CASE, eta_x=eta_x, **cfg_overrides)
        infra = build_infra(CASE, cfg)
        bank = generate_attack_only_bank(CASE, n_attack=N_ATTACK, c0=cfg.true_c, seed=SEED)
        rows = run_algorithms_parallel(ALGORITHMS, infra, bank, cfg, log=logger)
        for row in rows:
            row["case"] = CASE
            row["eta_x"] = eta_x
        all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(os.path.join(out_dir, "case57_budget_raw.csv"), index=False)

    summary_rows = []
    for eta_x in ETA_X_VALUES:
        sub = df_raw[df_raw["eta_x"] == eta_x]
        if len(sub) == 0:
            continue
        sr, sr_lo, sr_hi = bootstrap_ci(sub["detected"].values.astype(float))
        cost, cost_lo, cost_hi = bootstrap_ci(sub["cumulative_act_cost"].values)
        acc = float(sub["id_correct"].mean())
        summary_rows.append(
            {
                "eta_x": eta_x,
                "algorithm": "MS-HIDE",
                "sr_detect_mean": sr,
                "sr_detect_ci_low": sr_lo,
                "sr_detect_ci_high": sr_hi,
                "act_cost_mean": cost,
                "act_cost_ci_low": cost_lo,
                "act_cost_ci_high": cost_hi,
                "id_acc_mean": acc,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "case57_budget_summary.csv"), index=False)
    _plot(df_summary, out_dir)
    return df_summary


def _plot(df: pd.DataFrame, out_dir: str) -> None:
    plt = setup_matplotlib()
    algo = "MS-HIDE"
    sub = df[df["algorithm"] == algo].sort_values("eta_x")
    color = ALGO_COLORS[algo]
    ls = ALGO_LINESTYLES[algo]
    marker = ALGO_MARKERS[algo]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = sub["eta_x"].values

    ax1.plot(x, sub["sr_detect_mean"].values, color=color, linestyle=ls, marker=marker, label=algo)
    ax1.fill_between(x, sub["sr_detect_ci_low"].values, sub["sr_detect_ci_high"].values, color=color, alpha=0.15)
    ax2.plot(x, sub["act_cost_mean"].values, color=color, linestyle=ls, marker=marker, label=algo)
    ax2.fill_between(x, sub["act_cost_ci_low"].values, sub["act_cost_ci_high"].values, color=color, alpha=0.15)

    ax1.set_xlabel(r"Actuation budget $\eta_x$")
    ax1.set_ylabel("Detection success rate $SR_D$")
    ax1.set_title(r"(a) Detection rate vs. $\eta_x$")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r"Actuation budget $\eta_x$")
    ax2.set_ylabel(r"Mean cumulative cost $\bar{C}_u$")
    ax2.set_title(r"(b) Actuation cost vs. $\eta_x$")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_budget_tradeoff.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
