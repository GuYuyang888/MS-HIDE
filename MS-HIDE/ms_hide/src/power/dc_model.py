
from __future__ import annotations

from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray


class DCModel:

                                          
    _BUS_I = 0                              
    _BUS_TYPE = 1                         
    _BUS_PD = 2                              

                                          
    _GEN_BUS = 0                            
    _GEN_PG = 1                              
    _GEN_STATUS = 7                                    

                                             
    _BR_FBUS = 0                                   
    _BR_TBUS = 1                                 
    _BR_R = 2                           
    _BR_X = 3                          
    _BR_STATUS = 10                                  

    def __init__(self, case_data: Dict[str, Union[float, np.ndarray]]) -> None:
        bus_data: NDArray = case_data["bus"]
        gen_data: NDArray = case_data["gen"]
        branch_data: NDArray = case_data["branch"]
        self.baseMVA: float = float(case_data["baseMVA"])

                                                                            
                                                                        
                                                                            
        ext_bus_ids = bus_data[:, self._BUS_I].astype(np.int64)
        self.n_bus: int = len(ext_bus_ids)

                                                                        
        sort_order = np.argsort(ext_bus_ids)
        ext_bus_sorted = ext_bus_ids[sort_order]

                                      
        self.int2ext: NDArray[np.int64] = ext_bus_sorted.copy()
                                                          
        self._ext2int_map: Dict[int, int] = {
            int(ext): idx for idx, ext in enumerate(ext_bus_sorted)
        }
                                                                 
        max_ext = int(ext_bus_sorted.max())
        self.ext2int: NDArray[np.int64] = np.full(max_ext + 1, -1, dtype=np.int64)
        for ext, idx in self._ext2int_map.items():
            self.ext2int[ext] = idx

                                               
        bus_data = bus_data[sort_order]

                                                                            
                               
                                                                            
        bus_types = bus_data[:, self._BUS_TYPE].astype(int)
        slack_mask = bus_types == 3
        if not np.any(slack_mask):
            raise ValueError("No slack bus (type 3) found in bus data.")
                                                   
        self.slack_bus_idx: int = int(np.argmax(slack_mask))

                                                 
        self.n: int = self.n_bus - 1

                                                                            
                                   
                                                                            
        active_mask = branch_data[:, self._BR_STATUS].astype(int) == 1
        active_branches = branch_data[active_mask]
        self.n_l: int = len(active_branches)

        if self.n_l == 0:
            raise ValueError("No active branches found in branch data.")

                                                                            
                                                             
                                                                          
                                                                            
        x_vec = active_branches[:, self._BR_X]
        if np.any(x_vec == 0):
            raise ValueError(
                "Zero reactance found on active branches; DC model requires "
                "non-zero reactance for susceptance computation."
            )
                                                                                  
        tap = active_branches[:, 8].copy()
        tap[tap == 0] = 1.0
        self.b0: NDArray[np.float64] = 1.0 / (x_vec * tap)

                                                                            
                                                           
                                                                          
                                             
                                                                            
        self.A: NDArray[np.float64] = self._build_incidence_matrix(active_branches)

                                                                            
                                                   
                                                                            
        self.H_f0: NDArray[np.float64] = self.build_H_f(self.b0)
        self.H_p0: NDArray[np.float64] = self.build_H_p(self.b0)
        self.H0: NDArray[np.float64] = self.build_H(self.b0)

                                                                 
        self.m: int = self.H0.shape[0]
        assert self.m == self.n_l + self.n, (
            f"Measurement count mismatch: m={self.m}, n_l={self.n_l}, n={self.n}"
        )

                                                                            
                                                                    
                                                                            
        p_inj = self._compute_injection_vector(bus_data, gen_data)
        B_bus = self.H_p0                      
        self.theta0: NDArray[np.float64] = np.linalg.solve(B_bus, p_inj)
        if not np.all(np.isfinite(self.theta0)):
            raise ValueError("Solved theta0 contains non-finite values.")

                                                                            
                                                             
                                                                            
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.z_nom: NDArray[np.float64] = self.H0 @ self.theta0
        if not np.all(np.isfinite(self.z_nom)):
            raise ValueError("Nominal measurement z_nom contains non-finite values.")

                                                     
        self._active_branches = active_branches

                                                                        
                    
                                                                        

    def build_H_f(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        if b.shape != (self.n_l,):
            raise ValueError(
                f"Susceptance vector shape {b.shape} does not match "
                f"expected ({self.n_l},)."
            )
        if not np.all(np.isfinite(b)):
            raise ValueError("Susceptance vector contains non-finite values.")
                                                                               
        return b[:, None] * self.A

    def build_H_p(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        if b.shape != (self.n_l,):
            raise ValueError(
                f"Susceptance vector shape {b.shape} does not match "
                f"expected ({self.n_l},)."
            )
        if not np.all(np.isfinite(b)):
            raise ValueError("Susceptance vector contains non-finite values.")
        H_f = b[:, None] * self.A
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            H_p = self.A.T @ H_f
        if not np.all(np.isfinite(H_p)):
            raise ValueError("Computed H_p contains non-finite values.")
        return H_p

    def build_H(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        H_f = self.build_H_f(b)
        H_p = self.build_H_p(b)
        return np.vstack([H_f, H_p])

                                                                        
                     
                                                                        

    def _build_incidence_matrix(
        self, active_branches: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        A_full = np.zeros((self.n_l, self.n_bus), dtype=np.float64)

        for l_idx in range(self.n_l):
            fbus_ext = int(active_branches[l_idx, self._BR_FBUS])
            tbus_ext = int(active_branches[l_idx, self._BR_TBUS])

            fbus_int = self._ext2int_map[fbus_ext]
            tbus_int = self._ext2int_map[tbus_ext]

            A_full[l_idx, fbus_int] = +1.0
            A_full[l_idx, tbus_int] = -1.0

                                     
        non_slack_cols = [i for i in range(self.n_bus) if i != self.slack_bus_idx]
        return A_full[:, non_slack_cols]

    def _compute_injection_vector(
        self,
        bus_data: NDArray[np.float64],
        gen_data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
                                                
        p_bus = np.zeros(self.n_bus, dtype=np.float64)
        p_bus[:] = -bus_data[:, self._BUS_PD]

                        
        for g_idx in range(gen_data.shape[0]):
            status = int(gen_data[g_idx, self._GEN_STATUS])
            if status <= 0:
                continue                            
            gen_bus_ext = int(gen_data[g_idx, self._GEN_BUS])
            gen_bus_int = self._ext2int_map[gen_bus_ext]
            p_bus[gen_bus_int] += gen_data[g_idx, self._GEN_PG]

                             
        p_bus /= self.baseMVA

                                    
        non_slack = [i for i in range(self.n_bus) if i != self.slack_bus_idx]
        return p_bus[non_slack]

                                                                        
                 
                                                                        

    def __repr__(self) -> str:
        return (
            f"DCModel(n_bus={self.n_bus}, n={self.n}, n_l={self.n_l}, "
            f"m={self.m}, slack_bus_idx={self.slack_bus_idx})"
        )
