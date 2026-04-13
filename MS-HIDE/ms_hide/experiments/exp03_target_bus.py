from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ms_hide.experiments._common import ALGO_COLORS, ALGO_MARKERS, build_infra, bootstrap_ci, make_cfg, results_dir, run_algorithms_parallel, setup_matplotlib
from ms_hide.experiments.scenario_bank import generate_scenario_bank

logger = logging.getLogger(__name__)

CASE = "case57"
ALGORITHMS = ["MS-HIDE"]
N_EPISODES_PER_BUS = 100
SEED = 42
REPRESENTATIVE_BUSES = [32, 4, 15, 25, 38, 49, 2, 9, 17, 35, 5, 10, 23, 48, 3, 11, 14, 37, 8, 12]


def run(cfg_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if cfg_overrides is None:
        cfg_overrides = {}

    out_dir = results_dir("exp3_target_bus")
    cfg = make_cfg(CASE, **cfg_overrides)
    infra = build_infra(CASE, cfg)
    target_buses = sorted(REPRESENTATIVE_BUSES)
    bank = generate_scenario_bank(CASE, n_attack=N_EPISODES_PER_BUS, n_null=0, cfg=cfg, target_buses=target_buses, seed=SEED)
    rows = run_algorithms_parallel(ALGORITHMS, infra, bank, cfg, log=logger)
    target_by_episode = {int(entry["episode_id"]): int(entry.get("target_bus", 0)) for entry in bank}
    for row in rows:
        row["target_bus"] = target_by_episode.get(int(row.get("episode_id", -1)), 0)
        row["case"] = CASE

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(os.path.join(out_dir, "case57_targetbus_raw.csv"), index=False)

    bus_degree = _compute_bus_degree(infra.model)
    summary_rows = []
    for target_bus in target_buses:
        sub = df_raw[df_raw["target_bus"] == target_bus]
        if len(sub) == 0:
            continue
        sr, sr_lo, sr_hi = bootstrap_ci(sub["detected"].values.astype(float))
        acc, acc_lo, acc_hi = bootstrap_ci(sub["id_correct"].values.astype(float))
        delay = float(sub["tau_detect_censored"].mean())
        summary_rows.append(
            {
                "target_bus": target_bus,
                "degree": bus_degree.get(target_bus, 0),
                "algorithm": "MS-HIDE",
                "sr_detect_mean": sr,
                "sr_detect_ci_low": sr_lo,
                "sr_detect_ci_high": sr_hi,
                "id_acc_mean": acc,
                "id_acc_ci_low": acc_lo,
                "id_acc_ci_high": acc_hi,
                "delay_mean": delay,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "case57_targetbus_summary.csv"), index=False)
    _plot(df_summary, sorted(target_buses, key=lambda bus: (bus_degree.get(bus, 0), bus)), bus_degree, out_dir)
    return df_summary


def _compute_bus_degree(model) -> Dict[int, int]:
    degree: Dict[int, int] = defaultdict(int)
    slack = model.slack_bus_idx
    ext2int = model._ext2int_map
    branches = model._active_branches
    internal_degree: Dict[int, int] = defaultdict(int)
    for branch_index in range(model.n_l):
        fbus_ext = int(branches[branch_index, 0])
        tbus_ext = int(branches[branch_index, 1])
        internal_degree[ext2int[fbus_ext]] += 1
        internal_degree[ext2int[tbus_ext]] += 1
    for bus_int in range(model.n_bus):
        if bus_int == slack:
            continue
        state_idx = bus_int if bus_int < slack else bus_int - 1
        degree[state_idx + 1] = internal_degree.get(bus_int, 0)
    return degree


def _plot(df: pd.DataFrame, target_buses: List[int], bus_degree: Dict[int, int], out_dir: str) -> None:
    plt = setup_matplotlib()
    y_pos = np.arange(len(target_buses))
    color = ALGO_COLORS["MS-HIDE"]
    marker = ALGO_MARKERS["MS-HIDE"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.35 * len(target_buses) + 1.8), sharey=True)

    for ax, metric, lo_col, hi_col, title, xlabel in (
        (ax1, "sr_detect_mean", "sr_detect_ci_low", "sr_detect_ci_high", "(a) Detection success rate $SR_D$", "$SR_D$"),
        (ax2, "id_acc_mean", "id_acc_ci_low", "id_acc_ci_high", "(b) Identification accuracy $Acc_{ID}$", "$Acc_{ID}$"),
    ):
        sub = df.set_index("target_bus")
        values = [float(sub.loc[bus, metric]) for bus in target_buses]
        lows = [float(sub.loc[bus, lo_col]) for bus in target_buses]
        highs = [float(sub.loc[bus, hi_col]) for bus in target_buses]
        ax.errorbar(values, y_pos, xerr=[np.array(values) - np.array(lows), np.array(highs) - np.array(values)], fmt=marker, color=color, ecolor=color, elinewidth=1.0, capsize=2, markersize=5, linewidth=1.2)
        ax.plot(values, y_pos, color=color, linewidth=1.0, alpha=0.8)
        ax.set_xlim(-0.05, 1.08)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", alpha=0.25)

    labels = [f"Bus {bus}  (d={bus_degree.get(bus, '?')})" for bus in target_buses]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("Target bus")
    ax1.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_target_bus_profile.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
