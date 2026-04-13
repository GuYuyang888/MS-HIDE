
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GlobalConfig:

                                                                        
                         
                                                                        
    case_name: str = "default"
    matpower_file: str = ""                                             

                                                                        
                             
                                                                        
    T: int = 20                                                   
    horizon: int = 2                                             
    num_scenarios: int = 12                                  
    num_random_dirs: int = 30                                                       
    max_actions: int = 10                                                 

                                                                        
           
                                                                        
    sigma_rel: float = 0.01                                      
    sigma_min: float = 1e-4                                           

                                                                        
                              
                                                                        
    alpha_fa: float = 0.00256                                                         

                                                                        
                             
                                                                        
    eta_x: float = 0.20                                               
    hide_tol: float = 1e-8                                      
    zero_flow_tol: float = 0.01                                                  
    eps_denom: float = 1e-6                                             
    eps_prob: float = 1e-12                            
    eps_d: float = 1e-12                                                 
    detect_steepness: float = 4.0                                                     
    exploit_confidence_beta: float = 0.0                                                           
                                                                             
    robust_struct_undetect_tol: float = 1.0
    robust_rho_u: Optional[float] = None
    tau_obs: float = 1e-12                                              
    placement_max_attempts_per: int = 5000                                                            
    placement_max_attempts_per_strict: int = 800                                                     

                                                                        
                  
                                                                        
    lambda_H: float = 0.85                                             
                                           
                                                                            
                                                                
                                                                                 
    support_entropy_mode: str = "shannon"
    renyi_alpha: float = 2.0
    renyi_entropy_blend: float = 0.5
    xi_weight: float = 3.0                                                           
    xi_adaptive: bool = False                                                                           
    terminal_xi_bonus: float = 2.0                                                         
                                                                          
                                                
    terminal_entropy_bonus: float = 0.0
                                                                          
                                                                        
                                                                        
                                                
    terminal_margin_bonus: float = 0.0
                                                                           
                                                                         
                                                                         
    entropy_drop_bonus: float = 0.0
                                                                        
                                                                        
                                                                          
    cov_trace_enable: bool = False
    cov_trace_weight: float = 0.0
    cov_trace_terminal_bonus: float = 0.0
                                                           
                                                                         
                                                                        
                                                                          
    info_gain_bonus: float = 0.0
                                                             
                                                                                
                                         
    step1_xi_coupling: float = 0.0
                                                             
                                                    
                                                                
                                         
    step1_exploit_coupling: float = 0.0
                                                                 
                                                                 
                                         
                                                                       
                       
                               
                                         
    step1_dual_shared_coupling: float = 0.0
                                                                 
                                                           
                                                    
                                                                
                       
                               
                                         
    step1_phase_dual_coupling: float = 0.0
                                                                     
                                                                     
                                  
                                                                                          
                                            
                              
                                
                                         
    step1_temporal_coupling: float = 0.0
    step1_temporal_detect_ref: float = 0.55
    step1_temporal_detect_floor: float = 0.0
                                                     
                                           
                                                           
                                                 
    exploit_uncertainty_gate: bool = False
    exploit_gate_strength: float = 0.0
    exploit_gate_floor: float = 0.30
    rho_u: float = 0.05                                          
                                              
                                                                      
    rollout_topk_ratio: float = 0.20
    rollout_topk_min: int = 2
    rollout_topk_screen_scenarios: int = 1

                                                                        
            
                                                                        
    q_c: float = 1e-5                                                     
    prior_null: float = 0.20                                               
    prior_c_mean: float = 0.0                                    
    prior_c_var: float = 0.0025                                                

                                                                        
                         
                                                                        
    true_c: float = 0.04                                                  

                                                                        
          
                                                                        
    oracle_hiddenness: bool = True                                         
    strict_cycle_rules: bool = False                                                                    
    seed: int = 1

                                                                        
                                   
                                                                        
    def resolve_matpower_path(self, project_root: Optional[str] = None) -> str:
        if os.path.isabs(self.matpower_file):
            return self.matpower_file
        if project_root is None:
                                                               
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(project_root, self.matpower_file))
