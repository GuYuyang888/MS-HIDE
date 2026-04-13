
from __future__ import annotations

from .cfg_global import GlobalConfig


def make_config(**overrides) -> GlobalConfig:
    defaults = dict(
        case_name="case118",
        matpower_file="../matpower/data/case118.m",
                                                             
        zero_flow_tol=0.005,
        num_scenarios=16,
        num_random_dirs=500,
        max_actions=80,
                                                                  
                                                              
                                    
        detect_steepness=12.0,
                                                                 
                                                        
        xi_weight=5.0,
                                                                    
                                                                    
                                                                     
                                                                 
        alpha_fa=0.004,
                                                                      
                                                                  
                                                                   
                                                           
        true_c=0.06,
    )
    defaults.update(overrides)
    return GlobalConfig(**defaults)


                                                       
CFG_CASE118 = make_config()
