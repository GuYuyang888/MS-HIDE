
from __future__ import annotations

from .cfg_global import GlobalConfig


def make_config(**overrides) -> GlobalConfig:
    defaults = dict(
        case_name="case14",
        matpower_file="../matpower/data/case14.m",
                                                                               
        zero_flow_tol=0.001,
    )
    defaults.update(overrides)
    return GlobalConfig(**defaults)


                                                       
CFG_CASE14 = make_config()
