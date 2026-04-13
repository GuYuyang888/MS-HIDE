#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
import multiprocessing as _mp

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    _mp.set_start_method("spawn")
except RuntimeError:
    pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ms_hide.config.cfg_case14 import make_config as make_cfg14
from ms_hide.config.cfg_case57 import make_config as make_cfg57
from ms_hide.config.cfg_case118 import make_config as make_cfg118
from ms_hide.src.power.matpower_parser import parse_matpower_case

_EXPERIMENTS = [
    {
        "index": 1,
        "public_id": "exp01",
        "legacy_id": "exp01",
        "name": "Overall Comparison",
        "module": "ms_hide.experiments.exp01_overall",
    },
    {
        "index": 2,
        "public_id": "exp02",
        "legacy_id": "exp02",
        "name": "Attack Strength",
        "module": "ms_hide.experiments.exp02_attack_strength",
    },
    {
        "index": 3,
        "public_id": "exp03",
        "legacy_id": "exp03",
        "name": "Target Bus Profile",
        "module": "ms_hide.experiments.exp03_target_bus",
    },
    {
        "index": 4,
        "public_id": "exp04",
        "legacy_id": "exp05",
        "name": "Budget Tradeoff",
        "module": "ms_hide.experiments.exp05_budget",
    },
    {
        "index": 5,
        "public_id": "exp05",
        "legacy_id": "exp06",
        "name": "Noise Robustness",
        "module": "ms_hide.experiments.exp06_noise",
    },
    {
        "index": 6,
        "public_id": "exp06",
        "legacy_id": "exp07",
        "name": "Scalability",
        "module": "ms_hide.experiments.exp07_scalability",
    },
]

_SELECTOR_MAP = {}
for exp in _EXPERIMENTS:
    _SELECTOR_MAP[str(exp["index"])] = exp["index"]
    _SELECTOR_MAP[exp["public_id"]] = exp["index"]
    _SELECTOR_MAP[exp["legacy_id"]] = exp["index"]
    _SELECTOR_MAP[exp["name"].lower().replace(" ", "_")] = exp["index"]


def _preflight_matpower(logger: logging.Logger) -> None:
    for case_name, maker in (("case14", make_cfg14), ("case57", make_cfg57), ("case118", make_cfg118)):
        cfg = maker()
        path = cfg.resolve_matpower_path(_THIS_DIR)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MATPOWER case missing for {case_name}: {path}")
        parse_matpower_case(path)
        logger.info("MATPOWER preflight OK: %s -> %s", case_name, path)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _normalize_selection(raw_selectors: list[str]) -> list[dict]:
    if not raw_selectors:
        return list(_EXPERIMENTS)

    selected_indexes = []
    for raw in raw_selectors:
        key = raw.strip().lower()
        if key == "all":
            return list(_EXPERIMENTS)
        if key not in _SELECTOR_MAP:
            valid = ", ".join(sorted(_SELECTOR_MAP))
            raise ValueError(f"Unknown experiment selector '{raw}'. Valid selectors: {valid}")
        idx = _SELECTOR_MAP[key]
        if idx not in selected_indexes:
            selected_indexes.append(idx)
    return [exp for exp in _EXPERIMENTS if exp["index"] in selected_indexes]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MS-HIDE experiment runner")
    parser.add_argument("selectors", nargs="*", help="Experiment selectors: 1-6, exp01-exp06, exp01-exp07 aliases, or all")
    parser.add_argument("--dry-run", action="store_true", help="Only verify module imports")
    args = parser.parse_args(argv)

    _setup_logging()
    logger = logging.getLogger("MS-HIDE")

    selected = _normalize_selection(args.selectors)
    os.makedirs(os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    _preflight_matpower(logger)

    total_start = time.time()
    failed = False
    for position, exp in enumerate(selected, start=1):
        logger.info("")
        logger.info("=" * 60)
        logger.info("[%d/%d] %s", position, len(selected), exp["name"])
        logger.info("=" * 60)
        try:
            module = importlib.import_module(exp["module"])
            if args.dry_run:
                logger.info("Imported %s", exp["module"])
                continue
            if not hasattr(module, "run"):
                raise AttributeError(f"Module {exp['module']} has no run()")
            start = time.time()
            module.run()
            elapsed = time.time() - start
            logger.info("%s completed in %.1fs", exp["name"], elapsed)
        except Exception as exc:
            logger.error("%s failed: %s", exp["name"], exc)
            import traceback
            traceback.print_exc()
            failed = True

    elapsed_total = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Completed in %.1fs", elapsed_total)
    logger.info("Results saved to results/")
    logger.info("=" * 60)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
