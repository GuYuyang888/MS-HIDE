
from __future__ import annotations

from .cfg_global import GlobalConfig


def make_config(**overrides) -> GlobalConfig:
    defaults = dict(
        case_name="case57",
        matpower_file="../matpower/data/case57.m",
                                                              
                                                                          
        zero_flow_tol=0.003,
                                              
                                                                            
                                                                               
        num_scenarios=8,
        num_random_dirs=500,
        max_actions=80,
    )
    defaults.update(overrides)
    return GlobalConfig(**defaults)


                                                       
CFG_CASE57 = make_config()
