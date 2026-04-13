
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, ncx2

logger = logging.getLogger(__name__)


class BDDDetector:

    def __init__(self, nu: int, alpha_fa: float = 0.05) -> None:
        if nu < 1:
            raise ValueError(
                f"Residual subspace dimension nu must be >= 1, got {nu}."
            )
        if not (0.0 < alpha_fa < 1.0):
            raise ValueError(
                f"False-alarm rate alpha_fa must be in (0, 1), got {alpha_fa}."
            )

        self.nu: int = nu
        self.alpha_fa: float = alpha_fa

                                                           
        self.gamma_th: float = float(chi2.ppf(1.0 - alpha_fa, df=nu))

        logger.debug(
            "BDDDetector: nu=%d, alpha_fa=%.4f, gamma_th=%.6f",
            nu, alpha_fa, self.gamma_th,
        )

                                                                        
                           
                                                                        

    def compute_statistic(self, y_t: NDArray[np.float64]) -> float:
        y_t = np.asarray(y_t, dtype=np.float64).ravel()
        if y_t.size != self.nu:
            raise ValueError(
                f"y_t has {y_t.size} elements, expected {self.nu}."
            )
        return float(np.sum(y_t ** 2))

                                                                        
                                
                                                                        

    def detect(self, y_t: NDArray[np.float64]) -> bool:
        xi = self.compute_statistic(y_t)
        return bool(xi > self.gamma_th)

                                                                        
                                     
                                                                        

    @staticmethod
    def noncentrality(
        g_h: NDArray[np.float64],
        c_h: float,
    ) -> float:
        return float(c_h ** 2 * np.sum(g_h ** 2))

                                                                        
                                  
                                                                        

    def detection_probability(
        self,
        g_h: NDArray[np.float64],
        c_h: float,
    ) -> float:
        lambda_h = self.noncentrality(g_h, c_h)
        if lambda_h < 1e-300:
                                                                     
                                              
            return self.alpha_fa
        return float(1.0 - ncx2.cdf(self.gamma_th, df=self.nu, nc=lambda_h))

                                                                        
                                 
                                                                        

    def detection_probabilities(
        self,
        g_dict: Dict[int, NDArray[np.float64]],
        c_h: float,
    ) -> Dict[int, float]:
        return {
            h: self.detection_probability(g_h, c_h)
            for h, g_h in g_dict.items()
        }

                                                                        
                 
                                                                        

    def __repr__(self) -> str:
        return (
            f"BDDDetector(nu={self.nu}, alpha_fa={self.alpha_fa:.4f}, "
            f"gamma_th={self.gamma_th:.4f})"
        )
