
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


                                                                        
                
                                                                        

def generate_scenario_bank(
    case_name: str,
    n_attack: int = 200,
    n_null: int = 50,
    cfg: Any = None,
    attack_mags: Optional[Sequence[float]] = None,
    sigma_rels: Optional[Sequence[float]] = None,
    eta_xs: Optional[Sequence[float]] = None,
    target_buses: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)

                              
    default_c0 = float(getattr(cfg, "true_c", 0.04)) if cfg else 0.04
    default_sigma_rel = float(getattr(cfg, "sigma_rel", 0.01)) if cfg else 0.01
    default_eta_x = float(getattr(cfg, "eta_x", 0.20)) if cfg else 0.20

    bank: List[Dict[str, Any]] = []
    episode_id = 0

                                                                       
                                 
                                                                       
    if target_buses is not None:
                                
        for _ in range(n_null):
            bank.append(_make_entry(
                episode_id=episode_id,
                attack_flag=False,
                h_true=0,
                c_true=0.0,
                rng=rng,
                extra={"target_bus": 0},
            ))
            episode_id += 1

                                        
        for bus_h in target_buses:
            for _ in range(n_attack):
                sign = rng.choice([-1.0, 1.0])
                c_star = sign * default_c0
                bank.append(_make_entry(
                    episode_id=episode_id,
                    attack_flag=True,
                    h_true=int(bus_h),
                    c_true=float(c_star),
                    rng=rng,
                    extra={"target_bus": int(bus_h)},
                ))
                episode_id += 1

        return bank

                                                                       
                               
                                                                       
    sweep_values: List[Dict[str, Any]] = [{}]

    if attack_mags is not None:
        sweep_values = [
            {**sv, "attack_mag": float(m)}
            for sv in sweep_values for m in attack_mags
        ]
    if sigma_rels is not None:
        sweep_values = [
            {**sv, "sigma_rel": float(s)}
            for sv in sweep_values for s in sigma_rels
        ]
    if eta_xs is not None:
        sweep_values = [
            {**sv, "eta_x": float(e)}
            for sv in sweep_values for e in eta_xs
        ]

    for sweep_params in sweep_values:
        c0 = sweep_params.get("attack_mag", default_c0)

                         
        for _ in range(n_attack):
            sign = rng.choice([-1.0, 1.0])
            c_star = sign * c0
                                                                 
                                                                     
                                                   
                                                                  
                                                                  
            h_seed = int(rng.integers(0, 2**31))
            bank.append(_make_entry(
                episode_id=episode_id,
                attack_flag=True,
                h_true=-1,                                          
                c_true=float(c_star),
                rng=rng,
                extra={**sweep_params, "h_seed": h_seed},
            ))
            episode_id += 1

                       
        for _ in range(n_null):
            bank.append(_make_entry(
                episode_id=episode_id,
                attack_flag=False,
                h_true=0,
                c_true=0.0,
                rng=rng,
                extra=sweep_params,
            ))
            episode_id += 1

    return bank


def generate_attack_only_bank(
    case_name: str,
    n_attack: int = 200,
    c0: float = 0.04,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    bank: List[Dict[str, Any]] = []
    for i in range(n_attack):
        sign = rng.choice([-1.0, 1.0])
        c_star = sign * c0
        h_seed = int(rng.integers(0, 2**31))
        bank.append(_make_entry(
            episode_id=i,
            attack_flag=True,
            h_true=-1,
            c_true=float(c_star),
            rng=rng,
            extra={"h_seed": h_seed, "attack_mag": c0},
        ))
    return bank


                                                                        
                                                  
                                                                        

def resolve_h_true(scenario: Dict[str, Any], n: int, rng: np.random.Generator) -> int:
    h = scenario["h_true"]
    if h >= 1:
        return h
    if h == 0:
        return 0
                                          
    h_seed = scenario.get("h_seed", 0)
    h_rng = np.random.default_rng(h_seed)
    return int(h_rng.integers(1, n + 1))


                                                                        
             
                                                                        

def save_scenario_bank(
    bank: List[Dict[str, Any]],
    filepath: str,
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(bank, f, indent=2, default=_json_default)
    logger.info("Saved scenario bank (%d entries) to %s", len(bank), filepath)


def load_scenario_bank(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        bank = json.load(f)
    logger.info("Loaded scenario bank (%d entries) from %s", len(bank), filepath)
    return bank


                                                                        
                  
                                                                        

def _make_entry(
    episode_id: int,
    attack_flag: bool,
    h_true: int,
    c_true: float,
    rng: np.random.Generator,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "episode_id": episode_id,
        "attack_flag": bool(attack_flag),
        "h_true": int(h_true),
        "c_true": float(c_true),
        "noise_seed": int(rng.integers(0, 2**31)),
        "planner_seed": int(rng.integers(0, 2**31)),
    }
    if extra:
        entry.update(extra)
    return entry


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
