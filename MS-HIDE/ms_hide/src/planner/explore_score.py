
from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..utils.log_utils import entropy_normalized

if TYPE_CHECKING:
    pass                                                   

logger = logging.getLogger(__name__)
_G_DENSE_CACHE: Dict[Tuple[int, int, int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}


def _get_dense_g_matrix(
    g_dict: Dict[int, NDArray[np.float64]],
    n_hyp: int,
    nu: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    cache_key = (id(g_dict), n_hyp, nu, len(g_dict))
    cached = _G_DENSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    G = np.zeros((n_hyp, nu), dtype=np.float64)
    g_norm_sq = np.zeros(n_hyp, dtype=np.float64)
    for h, g_h in g_dict.items():
        if 0 < h < n_hyp:
            g = np.asarray(g_h, dtype=np.float64).ravel()
            if g.size == nu:
                G[h, :] = g
                g_norm_sq[h] = float(np.dot(g, g))

    _G_DENSE_CACHE[cache_key] = (G, g_norm_sq)
    return G, g_norm_sq


                                                                    
                        
                                                                    

def compute_support_entropy(belief, cfg) -> float:
    n_classes = len(belief.pi)
    eps_pi = float(getattr(cfg, "eps_prob", 1e-12))
    pi = np.asarray(belief.pi, dtype=np.float64)

                                                     
    h_shannon = entropy_normalized(pi, n_classes, eps_pi=eps_pi)

    mode = str(getattr(cfg, "support_entropy_mode", "shannon")).strip().lower()
    if mode == "shannon":
        return h_shannon

                                                                        
    alpha = float(getattr(cfg, "renyi_alpha", 2.0))
    if alpha <= 0.0:
        alpha = 2.0
    if abs(alpha - 1.0) < 1e-6:
        h_renyi = h_shannon
    else:
        pi_safe = np.maximum(pi, eps_pi)
        pi_safe /= np.sum(pi_safe)
        renyi_core = np.sum(np.power(pi_safe, alpha))
        if renyi_core <= eps_pi:
            h_renyi = 0.0
        else:
            h_renyi = float(
                np.log(renyi_core) / (1.0 - alpha) / np.log(max(n_classes, 2))
            )
            h_renyi = float(np.clip(h_renyi, 0.0, 1.0))

    if mode == "renyi":
        return h_renyi
    if mode == "hybrid":
        blend = float(np.clip(getattr(cfg, "renyi_entropy_blend", 0.5), 0.0, 1.0))
        return float((1.0 - blend) * h_shannon + blend * h_renyi)

    logger.debug("Unknown support_entropy_mode '%s'; fallback to shannon", mode)
    return h_shannon


                                                                    
                           
                                                                    

def compute_amplitude_variance(belief, cfg) -> float:
    P_h_0: float = float(cfg.prior_c_var)
    if P_h_0 <= 0.0:
                                                                         
        return 0.0

    pi = np.asarray(belief.pi, dtype=np.float64)
    P = np.asarray(belief.P, dtype=np.float64)
    if pi.size <= 1:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.dot(pi[1:], P[1:]) / P_h_0)


                                                                    
                          
                                                                    

def compute_uncertainty(
    belief,               
    cfg,
) -> Tuple[float, float, float]:
    h_supp = compute_support_entropy(belief, cfg)
    v_amp = compute_amplitude_variance(belief, cfg)

    lambda_H: float = float(cfg.lambda_H)
    xi = lambda_H * h_supp + (1.0 - lambda_H) * v_amp

    return xi, h_supp, v_amp


                                                                    
                                             
                                                                    

def predict_uncertainty(
    belief,               
    g_dict: Dict[int, NDArray[np.float64]],
    cfg,
    belief_updater=None,                 
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()

    S: int = int(getattr(cfg, "num_scenarios", 20))
    n_hyp = len(belief.pi)

                              
    nu: Optional[int] = None
    for g_h in g_dict.values():
        nu = g_h.shape[0]
        break
    if nu is None:
        xi_now, _, _ = compute_uncertainty(belief, cfg)
        return xi_now

                                                
    if belief_updater is not None:
        bp = belief_updater.predict(belief, cfg)
    else:
        bp = _approximate_predict_belief(belief, cfg)

    pi = np.asarray(bp.pi, dtype=np.float64)
    mu = np.asarray(bp.mu, dtype=np.float64)
    P = np.asarray(bp.P, dtype=np.float64)

                                                                        
    G, g_norm_sq = _get_dense_g_matrix(g_dict, n_hyp, nu)

                                                        
    pi_safe = np.maximum(pi, 0.0)
    pi_sum = float(np.sum(pi_safe))
    if pi_sum <= 1e-300:
        pi_safe = np.full(n_hyp, 1.0 / max(n_hyp, 1), dtype=np.float64)
    else:
        pi_safe /= pi_sum
    h_samples = rng.choice(n_hyp, size=S, p=pi_safe)
    c_samples = np.zeros(S, dtype=np.float64)
    nonnull = h_samples > 0
    if np.any(nonnull):
        h_nonnull = h_samples[nonnull]
        mu_sel = mu[h_nonnull]
        std_sel = np.sqrt(np.maximum(np.abs(P[h_nonnull]), 1e-30))
        c_samples[nonnull] = rng.normal(loc=mu_sel, scale=std_sel)

                                                     
    eps = rng.standard_normal((S, nu))
    y_batch = eps
    if np.any(nonnull):
        h_nonnull = h_samples[nonnull]
        active = g_norm_sq[h_nonnull] > 0
        if np.any(active):
            idx = np.flatnonzero(nonnull)[active]
            h_active = h_nonnull[active]
            y_batch[idx] += G[h_active, :] * c_samples[idx, None]

                                                                       
    try:
        from .batch_rollout import batch_belief_update_np, batch_uncertainty_np
    except ImportError:
        from ms_hide.src.planner.batch_rollout import batch_belief_update_np, batch_uncertainty_np

    pi_batch = np.broadcast_to(pi[None, :], (S, n_hyp)).copy()
    mu_batch = np.broadcast_to(mu[None, :], (S, n_hyp)).copy()
    P_batch = np.broadcast_to(P[None, :], (S, n_hyp)).copy()

    q_c = float(getattr(cfg, "q_c", 1e-5))
    pi_up, mu_up, P_up = batch_belief_update_np(
        pi_batch, mu_batch, P_batch, y_batch,
        G, g_norm_sq, q_c,
    )

    lambda_H = float(getattr(cfg, "lambda_H", 0.60))
    P0 = float(getattr(cfg, "prior_c_var", 0.0025))
    eps_prob = float(getattr(cfg, "eps_prob", 1e-12))
    xi_batch = batch_uncertainty_np(pi_up, P_up, lambda_H, P0, eps_prob)

    return float(np.mean(xi_batch))


def _copy_belief_like(belief):
    if hasattr(belief, "copy"):
        try:
            return belief.copy()
        except TypeError:
            pass
    return SimpleNamespace(
        pi=np.array(belief.pi, dtype=np.float64, copy=True),
        mu=np.array(belief.mu, dtype=np.float64, copy=True),
        P=np.array(belief.P, dtype=np.float64, copy=True),
    )


def _approximate_predict_belief(
    belief,
    cfg,
):
    q_c = float(getattr(cfg, "q_c", 1e-5))
    pi = np.array(belief.pi, dtype=np.float64, copy=True)
    mu = np.array(belief.mu, dtype=np.float64, copy=True)
    P = np.array(belief.P, dtype=np.float64, copy=True)
    if P.size > 1:
        P[1:] = P[1:] + q_c
    return SimpleNamespace(pi=pi, mu=mu, P=P)


                                                                    
                                              
                                                                    

def _approximate_belief_update(
    belief,               
    y_t: NDArray[np.float64],
    g_dict: Dict[int, NDArray[np.float64]],
    cfg,
):
    eps_prob: float = float(getattr(cfg, "eps_prob", 1e-12))
    n_hyp = len(belief.pi)
    nu = y_t.shape[0]

    y_norm_sq = float(np.dot(y_t, y_t))
    log_lik = np.full(n_hyp, -0.5 * y_norm_sq, dtype=np.float64)
    new_mu = np.array(belief.mu, dtype=np.float64)
    new_P = np.array(belief.P, dtype=np.float64)

    hyp_indices = [h for h in g_dict.keys() if 1 <= h < n_hyp]
    if hyp_indices:
        hs = np.asarray(hyp_indices, dtype=np.int64)
        g_stack = np.stack([g_dict[h] for h in hs], axis=0)
        mu_h = new_mu[hs]
        P_h = new_P[hs]
        g_norm_sq = np.einsum("ij,ij->i", g_stack, g_stack)

                                    
        innov = y_t[None, :] - g_stack * mu_h[:, None]

                                                           
                                 
        S_scalar = np.maximum(1.0 + P_h * g_norm_sq, 1e-15)

                                            
        g_innov = np.einsum("ij,ij->i", g_stack, innov)
        inv_innov = innov - (P_h / S_scalar)[:, None] * g_innov[:, None] * g_stack
        quad_form = np.einsum("ij,ij->i", innov, inv_innov)
        log_lik[hs] = -0.5 * (quad_form + np.log(S_scalar))

                                      
        K_scalar = P_h * g_norm_sq / S_scalar
        new_mu[hs] = mu_h + (P_h / S_scalar) * g_innov
        new_P[hs] = np.maximum(P_h * (1.0 - K_scalar), 1e-20)

                                 
    log_pi = np.log(np.maximum(belief.pi, eps_prob)) + log_lik
    log_pi_max = np.max(log_pi)
    log_pi_shifted = log_pi - log_pi_max
    pi_unnorm = np.exp(log_pi_shifted)
    pi_sum = np.sum(pi_unnorm)
    if pi_sum < 1e-300:
                                                   
        new_pi = np.ones(n_hyp, dtype=np.float64) / n_hyp
    else:
        new_pi = pi_unnorm / pi_sum

                            
    new_pi = np.clip(new_pi, eps_prob, 1.0)
    new_pi /= np.sum(new_pi)

    return SimpleNamespace(pi=new_pi, mu=new_mu, P=new_P)
