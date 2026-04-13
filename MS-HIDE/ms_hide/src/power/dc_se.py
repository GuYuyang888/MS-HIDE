
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SEResult:

    theta_hat: NDArray[np.floating]
    residual: NDArray[np.floating]
    gain: NDArray[np.floating]


@dataclass(frozen=True)
class ObservabilityResult:

    is_observable: bool
    rank_H: int
    rcond_val: float


class DCSE:

                                                                       
                  
                                                                       

    def __init__(
        self,
        z_nom: NDArray[np.floating],
        sigma_rel: float = 0.01,
        sigma_min: float = 1e-4,
    ) -> None:
        z_nom = np.asarray(z_nom, dtype=np.float64).ravel()
        if z_nom.size == 0:
            raise ValueError("z_nom must be a non-empty 1-D array.")
        if sigma_rel <= 0:
            raise ValueError(f"sigma_rel must be positive, got {sigma_rel}.")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}.")

        self.sigma_rel: float = sigma_rel
        self.sigma_min: float = sigma_min
        self.m: int = z_nom.size

        self.sigma, self.R_z, self.W, self.W_sqrt = self.build_noise_model(z_nom)

                                                                       
                            
                                                                       

    def build_noise_model(
        self, z_nom: NDArray[np.floating]
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        z_nom = np.asarray(z_nom, dtype=np.float64).ravel()
        sigma = np.maximum(self.sigma_rel * np.abs(z_nom), self.sigma_min)

        R_z = np.diag(sigma ** 2)
        W = np.diag(1.0 / sigma ** 2)
        W_sqrt = np.diag(1.0 / sigma)

        return sigma, R_z, W, W_sqrt

                                                                       
                                   
                                                                       

    def generate_measurement(
        self,
        H_t: NDArray[np.floating],
        theta_actual: NDArray[np.floating],
        attack_vector: Optional[NDArray[np.floating]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.floating]:
        H_t = np.asarray(H_t, dtype=np.float64)
        theta_actual = np.asarray(theta_actual, dtype=np.float64).ravel()

        if H_t.ndim != 2:
            raise ValueError(
                f"H_t must be 2-D, got shape {H_t.shape}."
            )
        m, n = H_t.shape
        if m != self.m:
            raise ValueError(
                f"H_t has {m} rows but noise model expects m={self.m}."
            )
        if theta_actual.size != n:
            raise ValueError(
                f"theta_actual has {theta_actual.size} elements but "
                f"H_t has {n} columns."
            )

                               
        z_t = H_t @ theta_actual

                          
        if attack_vector is not None:
            a_t = np.asarray(attack_vector, dtype=np.float64).ravel()
            if a_t.size != m:
                raise ValueError(
                    f"attack_vector has {a_t.size} elements but "
                    f"H_t has {m} rows."
                )
            z_t = z_t + a_t

                                    
        if rng is None:
            rng = np.random.default_rng()
        e_t = rng.normal(loc=0.0, scale=self.sigma)
        z_t = z_t + e_t

        return z_t

                                                                       
                              
                                                                       

    def estimate(
        self,
        z_t: NDArray[np.floating],
        H_t: NDArray[np.floating],
        W: Optional[NDArray[np.floating]] = None,
    ) -> SEResult:
        z_t = np.asarray(z_t, dtype=np.float64).ravel()
        H_t = np.asarray(H_t, dtype=np.float64)
        if W is None:
            W = self.W
        else:
            W = np.asarray(W, dtype=np.float64)

        if H_t.ndim != 2:
            raise ValueError(f"H_t must be 2-D, got shape {H_t.shape}.")
        m, n = H_t.shape
        if z_t.size != m:
            raise ValueError(
                f"z_t has {z_t.size} elements but H_t has {m} rows."
            )
        if W.shape != (m, m):
            raise ValueError(
                f"W has shape {W.shape}, expected ({m}, {m})."
            )
        if not np.all(np.isfinite(H_t)):
            raise ValueError("H_t contains non-finite values.")
        if not np.all(np.isfinite(z_t)):
            raise ValueError("z_t contains non-finite values.")
        if not np.all(np.isfinite(W)):
            raise ValueError("W contains non-finite values.")

                                    
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            HtW = H_t.T @ W                            
            HtWH = HtW @ H_t                           
        if not np.all(np.isfinite(HtW)) or not np.all(np.isfinite(HtWH)):
            raise ValueError("Normal-equation matrices contain non-finite values.")

                                                                  
                                                                        
                                              
                                                             
        try:
            K_t = np.linalg.solve(HtWH, HtW)               
        except np.linalg.LinAlgError:
            logger.error(
                "Normal-equation matrix H^T W H is singular; "
                "system is unobservable."
            )
            raise

        theta_hat = K_t @ z_t                             
        residual = z_t - H_t @ theta_hat                   

        return SEResult(theta_hat=theta_hat, residual=residual, gain=K_t)

                                                                       
                                    
                                                                       

    def check_observability(
        self,
        H_t: NDArray[np.floating],
        W: Optional[NDArray[np.floating]] = None,
        tau_obs: float = 1e-12,
    ) -> ObservabilityResult:
        H_t = np.asarray(H_t, dtype=np.float64)
        if W is None:
            W = self.W
        else:
            W = np.asarray(W, dtype=np.float64)

        if H_t.ndim != 2:
            raise ValueError(f"H_t must be 2-D, got shape {H_t.shape}.")
        m, n = H_t.shape

        if not np.all(np.isfinite(H_t)):
            return ObservabilityResult(
                is_observable=False,
                rank_H=0,
                rcond_val=0.0,
            )
        if not np.all(np.isfinite(W)):
            return ObservabilityResult(
                is_observable=False,
                rank_H=0,
                rcond_val=0.0,
            )

                              
        rank_H = int(np.linalg.matrix_rank(H_t))
        p1_ok = rank_H == n

                                                                   
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            HtWH = H_t.T @ W @ H_t                      
        if not np.all(np.isfinite(HtWH)):
            return ObservabilityResult(
                is_observable=False,
                rank_H=0,
                rcond_val=0.0,
            )
        sv = np.linalg.svd(HtWH, compute_uv=False)
                                          
        if sv[0] == 0.0:
            rcond_val = 0.0
        else:
            rcond_val = float(sv[-1] / sv[0])
        p2_ok = rcond_val >= tau_obs

        is_observable = p1_ok and p2_ok

        if not is_observable:
            logger.debug(
                "Observability check failed: rank(H)=%d (need %d), "
                "rcond=%.3e (need >= %.3e).",
                rank_H,
                n,
                rcond_val,
                tau_obs,
            )

        return ObservabilityResult(
            is_observable=is_observable,
            rank_H=rank_H,
            rcond_val=rcond_val,
        )
