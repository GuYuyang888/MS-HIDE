from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ms_hide.experiments._common import ALGO_COLORS, ALGO_LINESTYLES, ALGO_MARKERS, build_infra, bootstrap_ci, make_cfg, results_dir, run_algorithms_parallel, setup_matplotlib
from ms_hide.experiments.scenario_bank import generate_attack_only_bank

logger = logging.getLogger(__name__)

CASE = "case57"
ATTACK_MAGS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
ALGORITHMS = ["MS-HIDE"]
N_ATTACK = 200
SEED = 42


def run(cfg_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if cfg_overrides is None:
        cfg_overrides = {}

    out_dir = results_dir("exp2_attack_strength")
    all_rows: List[Dict[str, Any]] = []
    base_cfg = make_cfg(CASE, **cfg_overrides)
    infra = build_infra(CASE, base_cfg)

    for attack_mag in ATTACK_MAGS:
        logger.info("=" * 60)
        logger.info("Exp2: attack_mag = %.3f", attack_mag)
        logger.info("=" * 60)
        bank = generate_attack_only_bank(CASE, n_attack=N_ATTACK, c0=attack_mag, seed=SEED)
        cfg_mag_overrides = dict(cfg_overrides)
        cfg_mag_overrides["true_c"] = attack_mag
        cfg = make_cfg(CASE, **cfg_mag_overrides)
        rows = run_algorithms_parallel(ALGORITHMS, infra, bank, cfg, log=logger)
        for row in rows:
            row["case"] = CASE
            row["attack_mag"] = attack_mag
        all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(os.path.join(out_dir, "attack_strength_case57_raw.csv"), index=False)

    curve_rows = []
    for attack_mag in ATTACK_MAGS:
        sub = df_raw[df_raw["attack_mag"] == attack_mag]
        if len(sub) == 0:
            continue
        atk = sub[sub["attack_flag"] == True]
        if len(atk) == 0:
            continue
        sr, sr_lo, sr_hi = bootstrap_ci(atk["detected"].values.astype(float))
        dl, dl_lo, dl_hi = bootstrap_ci(atk["tau_detect_censored"].values.astype(float))
        curve_rows.append(
            {
                "attack_mag": attack_mag,
                "algorithm": "MS-HIDE",
                "sr_detect_mean": sr,
                "sr_detect_ci_low": sr_lo,
                "sr_detect_ci_high": sr_hi,
                "delay_mean": dl,
                "delay_ci_low": dl_lo,
                "delay_ci_high": dl_hi,
            }
        )

    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv(os.path.join(out_dir, "attack_strength_case57_curve.csv"), index=False)
    _plot(df_curve, out_dir)
    return df_curve


def _plot(df: pd.DataFrame, out_dir: str) -> None:
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    algo = "MS-HIDE"
    sub = df[df["algorithm"] == algo].sort_values("attack_mag")
    x = sub["attack_mag"].values
    color = ALGO_COLORS[algo]
    ls = ALGO_LINESTYLES[algo]
    marker = ALGO_MARKERS[algo]

    ax1.plot(x, sub["sr_detect_mean"].values, color=color, linestyle=ls, marker=marker, label=algo)
    ax1.fill_between(x, sub["sr_detect_ci_low"].values, sub["sr_detect_ci_high"].values, color=color, alpha=0.15)
    ax2.plot(x, sub["delay_mean"].values, color=color, linestyle=ls, marker=marker, label=algo)
    ax2.fill_between(x, sub["delay_ci_low"].values, sub["delay_ci_high"].values, color=color, alpha=0.15)

    ax1.set_xlabel("Attack magnitude $c_0$")
    ax1.set_ylabel("Detection success rate $SR_D$")
    ax1.set_title("(a) Detection rate vs. attack strength")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Attack magnitude $c_0$")
    ax2.set_ylabel(r"Censored mean delay $\bar{\tau}_D^{\mathrm{cens}}$")
    ax2.set_title("(b) Detection delay vs. attack strength")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_attack_strength.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
