
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..attack.fdi_attack import FDIAttackModel

logger = logging.getLogger(__name__)

                                                   
_RANK_TOL: float = 1e-10


class ResidualSubspace:

    def __init__(
        self,
        H_t: NDArray[np.float64],
        W_sqrt: NDArray[np.float64],
        H0: NDArray[np.float64],
        rank_tol: float = _RANK_TOL,
    ) -> None:
        H_t = np.asarray(H_t, dtype=np.float64)
        W_sqrt = np.asarray(W_sqrt, dtype=np.float64)
        H0 = np.asarray(H0, dtype=np.float64)

        if H_t.ndim != 2:
            raise ValueError(f"H_t must be 2-D, got shape {H_t.shape}.")
        self.m, self.n = H_t.shape

        if W_sqrt.shape != (self.m, self.m):
            raise ValueError(
                f"W_sqrt shape {W_sqrt.shape} does not match "
                f"expected ({self.m}, {self.m})."
            )
        if H0.shape != (self.m, self.n):
            raise ValueError(
                f"H0 shape {H0.shape} does not match "
                f"expected ({self.m}, {self.n})."
            )
        if not np.all(np.isfinite(H_t)):
            raise ValueError("H_t contains non-finite values.")
        if not np.all(np.isfinite(W_sqrt)):
            raise ValueError("W_sqrt contains non-finite values.")
        if not np.all(np.isfinite(H0)):
            raise ValueError("H0 contains non-finite values.")

        self._W_sqrt = W_sqrt
        self._H0 = H0

                                  
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.H_bar: NDArray[np.float64] = W_sqrt @ H_t
        if not np.all(np.isfinite(self.H_bar)):
            raise ValueError("Whitened matrix H_bar contains non-finite values.")

                                             
        U, s, _ = np.linalg.svd(self.H_bar, full_matrices=True)

                        
        if s.size == 0:
            rank = 0
        else:
            rank = int(np.sum(s > rank_tol * s[0]))

        self.nu: int = self.m - rank

        if self.nu <= 0:
            raise ValueError(
                f"Residual subspace dimension nu={self.nu} is non-positive. "
                f"The measurement matrix is full-row-rank (rank={rank}, m={self.m}). "
                f"BDD is impossible with this configuration."
            )

                                                                       
                                           
        self.Q_perp: NDArray[np.float64] = U[:, rank:]

        logger.debug(
            "ResidualSubspace: m=%d, n=%d, rank=%d, nu=%d",
            self.m, self.n, rank, self.nu,
        )

                                                                        
                       
                                                                        

    def project(self, z_t: NDArray[np.float64]) -> NDArray[np.float64]:
        z_t = np.asarray(z_t, dtype=np.float64).ravel()
        if z_t.size != self.m:
            raise ValueError(
                f"z_t has {z_t.size} elements, expected {self.m}."
            )
        if not np.all(np.isfinite(z_t)):
            raise ValueError("z_t contains non-finite values.")
                                        
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            z_bar = self._W_sqrt @ z_t
        if not np.all(np.isfinite(z_bar)):
            raise ValueError("Whitened observation z_bar contains non-finite values.")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            y_t = self.Q_perp.T @ z_bar
        if not np.all(np.isfinite(y_t)):
            raise ValueError("Projected observation y_t contains non-finite values.")
        return y_t

                                                                        
                               
                                                                        

    def compute_g_h(
        self,
        h: int,
        attack_model: FDIAttackModel,
    ) -> NDArray[np.float64]:
        if h == 0:
            raise ValueError(
                "Sensitivity vector is undefined for the null hypothesis (h=0)."
            )

                                                        
        d_h = attack_model.get_attack_direction(h)
        if not np.all(np.isfinite(d_h)):
            raise ValueError("Attack direction contains non-finite values.")

                                       
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            wd = self._W_sqrt @ d_h
            g_h = self.Q_perp.T @ wd
        if not np.all(np.isfinite(g_h)):
            raise ValueError("Computed g_h contains non-finite values.")
        return g_h

                                                                        
                                                  
                                                                        

    def compute_all_g(
        self,
        attack_model: FDIAttackModel,
    ) -> Dict[int, NDArray[np.float64]]:
        g_dict: Dict[int, NDArray[np.float64]] = {}
        if not np.all(np.isfinite(self._H0)):
            raise ValueError("H0 contains non-finite values.")

                                                                     
                                                      
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            G_matrix = self.Q_perp.T @ self._W_sqrt @ self._H0
        if not np.all(np.isfinite(G_matrix)):
            raise ValueError("Computed G_matrix contains non-finite values.")

        for hyp in attack_model.hypotheses:
            if hyp.id == 0:
                continue                        
                                                           
                                                
            g_dict[hyp.id] = G_matrix[:, hyp.id - 1].copy()

        return g_dict

                                                                        
                 
                                                                        

    def __repr__(self) -> str:
        return (
            f"ResidualSubspace(m={self.m}, n={self.n}, nu={self.nu})"
        )
