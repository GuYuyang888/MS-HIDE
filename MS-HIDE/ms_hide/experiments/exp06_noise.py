from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ms_hide.experiments._common import ALGO_COLORS, ALGO_LINESTYLES, ALGO_MARKERS, build_infra, bootstrap_ci, make_cfg, results_dir, run_algorithms_parallel, setup_matplotlib
from ms_hide.experiments.scenario_bank import generate_scenario_bank

logger = logging.getLogger(__name__)

CASE = "case57"
SIGMA_RELS = [0.005, 0.01, 0.02, 0.03]
ALGORITHMS = ["MS-HIDE"]
N_ATTACK = 160
N_NULL = 40
SEED = 42


def run(cfg_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if cfg_overrides is None:
        cfg_overrides = {}

    out_dir = results_dir("exp6_noise_robustness")
    all_rows: List[Dict[str, Any]] = []
    for sigma_rel in SIGMA_RELS:
        logger.info("=" * 60)
        logger.info("Exp6: sigma_rel = %.4f", sigma_rel)
        logger.info("=" * 60)
        cfg = make_cfg(CASE, sigma_rel=sigma_rel, **cfg_overrides)
        infra = build_infra(CASE, cfg)
        bank = generate_scenario_bank(CASE, n_attack=N_ATTACK, n_null=N_NULL, cfg=cfg, seed=SEED)
        rows = run_algorithms_parallel(ALGORITHMS, infra, bank, cfg, log=logger)
        for row in rows:
            row["case"] = CASE
            row["sigma_rel"] = sigma_rel
        all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(os.path.join(out_dir, "case57_noise_raw.csv"), index=False)

    summary_rows = []
    for sigma_rel in SIGMA_RELS:
        sub = df_raw[df_raw["sigma_rel"] == sigma_rel]
        if len(sub) == 0:
            continue
        atk = sub[sub["attack_flag"] == True]
        null = sub[sub["attack_flag"] == False]
        sr, sr_lo, sr_hi = bootstrap_ci(atk["detected"].values.astype(float)) if len(atk) else (0.0, 0.0, 0.0)
        far, far_lo, far_hi = bootstrap_ci(null["false_alarm"].values.astype(float)) if len(null) else (0.0, 0.0, 0.0)
        delay = float(atk["tau_detect_censored"].mean()) if len(atk) else 0.0
        summary_rows.append(
            {
                "sigma_rel": sigma_rel,
                "algorithm": "MS-HIDE",
                "sr_detect_mean": sr,
                "sr_detect_ci_low": sr_lo,
                "sr_detect_ci_high": sr_hi,
                "far_mean": far,
                "far_ci_low": far_lo,
                "far_ci_high": far_hi,
                "delay_mean": delay,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "case57_noise_summary.csv"), index=False)
    _plot(df_summary, out_dir)
    return df_summary


def _plot(df: pd.DataFrame, out_dir: str) -> None:
    plt = setup_matplotlib()
    algo = "MS-HIDE"
    sub = df[df["algorithm"] == algo].sort_values("sigma_rel")
    color = ALGO_COLORS[algo]
    ls = ALGO_LINESTYLES[algo]
    marker = ALGO_MARKERS[algo]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = sub["sigma_rel"].values

    ax1.plot(x, sub["sr_detect_mean"].values, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=7, label=algo)
    ax1.fill_between(x, sub["sr_detect_ci_low"].values, sub["sr_detect_ci_high"].values, color=color, alpha=0.12)
    ax2.plot(x, sub["delay_mean"].values, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=7, label=algo)

    ax1.set_xlabel(r"Measurement noise $\sigma_{\mathrm{rel}}$", fontsize=11)
    ax1.set_ylabel(r"Detection success rate $SR_D$", fontsize=11)
    ax1.set_title(r"(a) Detection rate vs. $\sigma_{\mathrm{rel}}$", fontsize=11.5)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=8.5, edgecolor="0.7")
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    far_text = ", ".join(f"{sigma:.3f}:{value:.3f}" for sigma, value in zip(sub["sigma_rel"].values, sub["far_mean"].values))
    ax1.text(0.98, 0.02, f"FAR\n({far_text})", transform=ax1.transAxes, fontsize=7, ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax2.set_xlabel(r"Measurement noise $\sigma_{\mathrm{rel}}$", fontsize=11)
    ax2.set_ylabel(r"Censored mean delay $\bar{\tau}_D^{\mathrm{cens}}$", fontsize=11)
    ax2.set_title(r"(b) Detection delay vs. $\sigma_{\mathrm{rel}}$", fontsize=11.5)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5, edgecolor="0.7")
    ax2.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_noise_robustness.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
