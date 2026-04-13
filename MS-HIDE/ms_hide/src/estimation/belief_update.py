
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..utils.log_utils import log_normalize

logger = logging.getLogger(__name__)

                                                               
_EPS_PROB: float = 1e-300
                                                                    
_EPS_VAR: float = 1e-30
                       
_LOG_2PI: float = float(np.log(2.0 * np.pi))


                                                                        
                        
                                                                        

@dataclass
class BeliefState:

    pi: NDArray[np.float64]
    mu: NDArray[np.float64]
    P: NDArray[np.float64]

    def copy(self) -> BeliefState:
        return BeliefState(
            pi=self.pi.copy(),
            mu=self.mu.copy(),
            P=self.P.copy(),
        )


                                                                        
                
                                                                        

class BeliefUpdater:

    def __init__(self, n_hypotheses: int, cfg: Any) -> None:
        if n_hypotheses < 2:
            raise ValueError(
                f"n_hypotheses must be >= 2 (null + at least one attack), "
                f"got {n_hypotheses}."
            )
        self.n_hypotheses: int = n_hypotheses
        self._eps_prob: float = getattr(cfg, "eps_prob", _EPS_PROB)
        self._eps_var: float = getattr(cfg, "eps_var", _EPS_VAR)
                                                                        
                                                                           
                                                                      
        self._g_cache: Dict[
            int,
            Tuple[
                NDArray[np.int64],
                NDArray[np.float64],
                NDArray[np.float64],
                int,
            ],
        ] = {}

                                                                        
                                 
                                                                        

    def initialize(self, cfg: Any) -> BeliefState:
        n_hyp = self.n_hypotheses
        prior_null: float = float(cfg.prior_null)

                           
        pi = np.empty(n_hyp, dtype=np.float64)
        pi[0] = prior_null
        if n_hyp > 1:
            pi[1:] = (1.0 - prior_null) / (n_hyp - 1)

                           
        mu = np.zeros(n_hyp, dtype=np.float64)
        mu0_val: float = float(getattr(cfg, "prior_c_mean", getattr(cfg, "mu0", 0.0)))
        mu[1:] = mu0_val                                        

                          
        P = np.zeros(n_hyp, dtype=np.float64)
        P0_val: float = float(getattr(cfg, "prior_c_var", getattr(cfg, "P0", 0.0025)))
        P[1:] = P0_val                         

        logger.debug(
            "BeliefUpdater.initialize: n_hyp=%d, pi[0]=%.4f, mu0=%.4f, P0=%.4f",
            n_hyp, prior_null, mu0_val, P0_val,
        )

        return BeliefState(pi=pi, mu=mu, P=P)

                                                                        
                                
                                                                        

    def predict(self, belief: BeliefState, cfg: Any) -> BeliefState:
        q_c: float = float(cfg.q_c)
        pred = belief.copy()
                                  
                                                                  
        pred.P[1:] = pred.P[1:] + q_c
                                    
        np.maximum(pred.P, self._eps_var, out=pred.P)
        return pred

                                                                        
                                             
                                                                        

    def update(
        self,
        belief: BeliefState,
        y_t: NDArray[np.float64],
        g_dict: Dict[int, NDArray[np.float64]],
        cfg: Any,
    ) -> BeliefState:
        y_t = np.asarray(y_t, dtype=np.float64).ravel()
        nu: int = y_t.size
        n_hyp: int = self.n_hypotheses

                       
        pi_new = np.empty(n_hyp, dtype=np.float64)
                                                                          
                               
        mu_new = belief.mu.copy()
        P_new = belief.P.copy()

                                                       
        log_weights = np.full(n_hyp, -np.inf, dtype=np.float64)

                                         
        log_L_0 = self._log_likelihood_null(y_t, nu)
        log_weights[0] = np.log(max(belief.pi[0], self._eps_prob)) + log_L_0
        mu_new[0] = 0.0           
        P_new[0] = 0.0             

                                                                        
        idx_arr, G, g_norm_sq = self._get_g_matrix(g_dict, nu)
        if idx_arr.size > 0:
            mu_prior = belief.mu[idx_arr]
            P_prior = belief.P[idx_arr]

                                               
            denom = 1.0 + P_prior * g_norm_sq
            denom = np.maximum(denom, self._eps_var)

            innov = y_t[None, :] - G * mu_prior[:, None]
            innov_norm_sq = np.einsum("ij,ij->i", innov, innov)
            gT_innov = np.einsum("ij,ij->i", G, innov)

                                                                
            log_det_S = np.log(denom)
            quad_form = innov_norm_sq - (P_prior / denom) * (gT_innov ** 2)
            log_L = -0.5 * (nu * _LOG_2PI + log_det_S + quad_form)
            log_weights[idx_arr] = np.log(np.maximum(belief.pi[idx_arr], self._eps_prob)) + log_L

                                                      
            k_scale = P_prior / denom
            mu_new[idx_arr] = mu_prior + k_scale * gT_innov
            P_new[idx_arr] = P_prior / denom

                                                      
        _, pi_new = log_normalize(log_weights)

                                                     
        pi_new = np.maximum(pi_new, self._eps_prob)
        pi_new /= pi_new.sum()

                         
        np.maximum(P_new, self._eps_var, out=P_new)
        P_new[0] = 0.0                                

        return BeliefState(pi=pi_new, mu=mu_new, P=P_new)

    def _get_g_matrix(
        self,
        g_dict: Dict[int, NDArray[np.float64]],
        nu: int,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        key = id(g_dict)
        cached = self._g_cache.get(key)
        if cached is not None:
            idx_arr, G, g_norm_sq, cached_nu = cached
            if cached_nu == nu:
                return idx_arr, G, g_norm_sq

        idxs = []
        rows = []
        for h, g_h in g_dict.items():
            if h <= 0 or h >= self.n_hypotheses:
                continue
            g = np.asarray(g_h, dtype=np.float64).ravel()
            if g.size != nu or not np.all(np.isfinite(g)):
                continue
            idxs.append(int(h))
            rows.append(g)

        if idxs:
            order = np.argsort(np.asarray(idxs, dtype=np.int64))
            idx_arr = np.asarray(idxs, dtype=np.int64)[order]
            G = np.vstack(rows).astype(np.float64, copy=False)[order, :]
            g_norm_sq = np.einsum("ij,ij->i", G, G)
        else:
            idx_arr = np.empty(0, dtype=np.int64)
            G = np.empty((0, nu), dtype=np.float64)
            g_norm_sq = np.empty(0, dtype=np.float64)

        self._g_cache[key] = (idx_arr, G, g_norm_sq, nu)
        return idx_arr, G, g_norm_sq

                                                                        
                                           
                                                                        

    @staticmethod
    def _log_likelihood_null(
        y_t: NDArray[np.float64],
        nu: int,
    ) -> float:
        return -0.5 * (nu * _LOG_2PI + float(np.dot(y_t, y_t)))

                                                                        
                                               
                                                                        

    @staticmethod
    def _log_likelihood_h(
        y_t: NDArray[np.float64],
        g_h: NDArray[np.float64],
        mu_h_prior: float,
        P_h_prior: float,
        nu: int,
    ) -> float:
        g_norm_sq = float(np.dot(g_h, g_h))
        denom = 1.0 + P_h_prior * g_norm_sq

                           
        innov = y_t - g_h * mu_h_prior               

                                                   
        log_det_S = np.log(denom)

                                                          
                                                                            
        gT_innov = float(np.dot(g_h, innov))
        Sinv_innov = innov - (P_h_prior / denom) * g_h * gT_innov
        quad_form = float(innov @ Sinv_innov)

        return -0.5 * (nu * _LOG_2PI + log_det_S + quad_form)

                                                                        
                 
                                                                        

    def __repr__(self) -> str:
        return f"BeliefUpdater(n_hypotheses={self.n_hypotheses})"
