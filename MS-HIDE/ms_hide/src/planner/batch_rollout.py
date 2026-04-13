
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

                                                                      
                                                          
_USE_TORCH = False
_DEVICE = None
_TORCH_CHECKED = False


def _check_torch():
    global _USE_TORCH, _DEVICE, _TORCH_CHECKED
    if _TORCH_CHECKED:
        return
    _TORCH_CHECKED = True
    try:
        import torch
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if mps_available:
                                                                          
                                                                           
            if os.environ.get("MSHIDE_ENABLE_MPS", "0") == "1":
                _USE_TORCH = True
                _DEVICE = torch.device("mps")
                logger.info("batch_rollout: using PyTorch MPS")
            else:
                logger.info(
                    "batch_rollout: MPS detected; using numpy by default "
                    "(set MSHIDE_ENABLE_MPS=1 to enable MPS)"
                )
        elif torch.cuda.is_available():
            _USE_TORCH = True
            _DEVICE = torch.device("cuda")
            logger.info("batch_rollout: using PyTorch CUDA (%s)", torch.cuda.get_device_name(0))
        else:
            logger.info("batch_rollout: PyTorch available but no GPU; using numpy")
    except ImportError:
        logger.info("batch_rollout: PyTorch not available; using numpy")

_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS_PROB = 1e-300
_EPS_VAR = 1e-30


def _to_tensor(arr):
    return torch.as_tensor(arr, dtype=torch.float64, device=_DEVICE)


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


                                                                        
                       
                                                                        

def precompute_G_tensor(
    g_dicts: List[Dict[int, NDArray[np.float64]]],
    n_hyp: int,
    nu: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    L = len(g_dicts)
    G = np.zeros((L, n_hyp, nu), dtype=np.float64)
    g_norm_sq = np.zeros((L, n_hyp), dtype=np.float64)

    for i, gd in enumerate(g_dicts):
        for h, g_h in gd.items():
            if 0 < h < n_hyp:
                g = np.asarray(g_h, dtype=np.float64).ravel()
                if g.size == nu:
                    G[i, h, :] = g
                    g_norm_sq[i, h] = float(np.dot(g, g))

    return G, g_norm_sq


                                                                        
                             
                                                                        

def batch_belief_update_np(
    pi_prior: NDArray[np.float64],               
    mu_prior: NDArray[np.float64],               
    P_prior: NDArray[np.float64],                
    y: NDArray[np.float64],                   
    G_action: NDArray[np.float64],                                            
    g_norm_sq: NDArray[np.float64],                                    
    q_c: float,
) -> Tuple[NDArray, NDArray, NDArray]:
    B, n_hyp = pi_prior.shape
    nu = y.shape[1]

                                               
    P_pred = P_prior.copy()
    P_pred[:, 1:] += q_c
    np.maximum(P_pred, _EPS_VAR, out=P_pred)

                                       
    if G_action.ndim == 2:
        G_action = np.broadcast_to(G_action[None, :, :], (B, n_hyp, nu))
    if g_norm_sq.ndim == 1:
        g_norm_sq = np.broadcast_to(g_norm_sq[None, :], (B, n_hyp))

                                                   
    denom = 1.0 + P_pred * g_norm_sq
    np.maximum(denom, _EPS_VAR, out=denom)

                                                    
    g_dot_y = np.einsum("bhj,bj->bh", G_action, y)

                                                                     
    g_dot_innov = g_dot_y - mu_prior * g_norm_sq              

                                 
    y_norm_sq = np.einsum("bj,bj->b", y, y)        

                                                 
                                                   
    innov_norm_sq = (y_norm_sq[:, None]
                     - 2.0 * mu_prior * g_dot_y
                     + mu_prior ** 2 * g_norm_sq)              

                                                                                                
    quad_form = innov_norm_sq - (P_pred / denom) * (g_dot_innov ** 2)
    log_lik = -0.5 * (np.log(denom) + quad_form)              

                                    
    log_lik[:, 0] = -0.5 * y_norm_sq                 

                                                
                                                                                          

                 
    log_pi = np.log(np.maximum(pi_prior, _EPS_PROB)) + log_lik
                             
    log_pi_max = np.max(log_pi, axis=1, keepdims=True)
    log_pi_shifted = log_pi - log_pi_max
    pi_unnorm = np.exp(log_pi_shifted)
    pi_sum = np.sum(pi_unnorm, axis=1, keepdims=True)
    pi_new = pi_unnorm / np.maximum(pi_sum, _EPS_PROB)
    np.clip(pi_new, _EPS_PROB, 1.0, out=pi_new)
    pi_new /= np.sum(pi_new, axis=1, keepdims=True)

                                           
    k_scale = P_pred / denom              
    mu_new = mu_prior.copy()
    mu_new[:, 1:] = mu_prior[:, 1:] + k_scale[:, 1:] * g_dot_innov[:, 1:]

    P_new = P_prior.copy()
    P_new[:, 1:] = P_pred[:, 1:] / denom[:, 1:]
    np.maximum(P_new, _EPS_VAR, out=P_new)
    P_new[:, 0] = 0.0

    return pi_new, mu_new, P_new


                                                                        
                                   
                                                                        

def batch_belief_update_torch(
    pi_prior,               
    mu_prior,               
    P_prior,                
    y,                   
    G_action,                                  
    g_norm_sq,                          
    q_c: float,
):
    B, n_hyp = pi_prior.shape
    nu = y.shape[1]

    P_pred = P_prior.clone()
    P_pred[:, 1:] += q_c
    P_pred.clamp_(min=_EPS_VAR)

    if G_action.dim() == 2:
        G_action = G_action.unsqueeze(0).expand(B, -1, -1)
    if g_norm_sq.dim() == 1:
        g_norm_sq = g_norm_sq.unsqueeze(0).expand(B, -1)

    denom = 1.0 + P_pred * g_norm_sq
    denom.clamp_(min=_EPS_VAR)

    g_dot_y = torch.einsum("bhj,bj->bh", G_action, y)
    g_dot_innov = g_dot_y - mu_prior * g_norm_sq

    y_norm_sq = torch.einsum("bj,bj->b", y, y)

    innov_norm_sq = (y_norm_sq.unsqueeze(1)
                     - 2.0 * mu_prior * g_dot_y
                     + mu_prior ** 2 * g_norm_sq)

    quad_form = innov_norm_sq - (P_pred / denom) * (g_dot_innov ** 2)
    log_lik = -0.5 * (torch.log(denom) + quad_form)
    log_lik[:, 0] = -0.5 * y_norm_sq

    log_pi = torch.log(pi_prior.clamp(min=_EPS_PROB)) + log_lik
    pi_new = torch.softmax(log_pi, dim=1)
    pi_new.clamp_(min=_EPS_PROB)
    pi_new /= pi_new.sum(dim=1, keepdim=True)

    k_scale = P_pred / denom
    mu_new = mu_prior.clone()
    mu_new[:, 1:] = mu_prior[:, 1:] + k_scale[:, 1:] * g_dot_innov[:, 1:]

    P_new = P_prior.clone()
    P_new[:, 1:] = P_pred[:, 1:] / denom[:, 1:]
    P_new.clamp_(min=_EPS_VAR)
    P_new[:, 0] = 0.0

    return pi_new, mu_new, P_new


                                                                        
                               
                                                                        

def batch_uncertainty_np(
    pi: NDArray[np.float64],               
    P: NDArray[np.float64],                
    lambda_H: float,
    P0: float,
    eps_prob: float = 1e-12,
    support_entropy_mode: str = "shannon",
    renyi_alpha: float = 2.0,
    renyi_entropy_blend: float = 0.5,
) -> NDArray[np.float64]:
    n_hyp = pi.shape[1]
    h_supp = support_entropy_batch_np(
        pi,
        eps_prob=eps_prob,
        mode=support_entropy_mode,
        renyi_alpha=renyi_alpha,
        renyi_entropy_blend=renyi_entropy_blend,
    )

                              
    if P0 > 0:
        v_amp = np.sum(pi[:, 1:] * P[:, 1:], axis=1) / P0        
    else:
        v_amp = np.zeros(pi.shape[0], dtype=np.float64)

    return lambda_H * h_supp + (1.0 - lambda_H) * v_amp


def batch_uncertainty_torch(
    pi,
    P,
    lambda_H: float,
    P0: float,
    eps_prob: float = 1e-12,
    support_entropy_mode: str = "shannon",
    renyi_alpha: float = 2.0,
    renyi_entropy_blend: float = 0.5,
):
    h_supp = support_entropy_batch_torch(
        pi,
        eps_prob=eps_prob,
        mode=support_entropy_mode,
        renyi_alpha=renyi_alpha,
        renyi_entropy_blend=renyi_entropy_blend,
    )

    if P0 > 0:
        v_amp = (pi[:, 1:] * P[:, 1:]).sum(dim=1) / P0
    else:
        v_amp = torch.zeros(pi.shape[0], dtype=torch.float64, device=pi.device)

    return lambda_H * h_supp + (1.0 - lambda_H) * v_amp


def support_entropy_batch_np(
    pi: NDArray[np.float64],              
    eps_prob: float = 1e-12,
    mode: str = "shannon",
    renyi_alpha: float = 2.0,
    renyi_entropy_blend: float = 0.5,
) -> NDArray[np.float64]:
    n_hyp = pi.shape[1]
    log_n = np.log(max(n_hyp, 2))
    pi_safe = np.maximum(pi, eps_prob)
    pi_safe /= np.sum(pi_safe, axis=1, keepdims=True)
    h_shannon = -np.sum(pi_safe * np.log(pi_safe), axis=1) / log_n        

    mode = str(mode).strip().lower()
    if mode == "shannon":
        return h_shannon

    alpha = float(renyi_alpha)
    if alpha <= 0.0:
        alpha = 2.0
    if abs(alpha - 1.0) < 1e-6:
        h_renyi = h_shannon
    else:
        renyi_core = np.sum(np.power(pi_safe, alpha), axis=1)
        h_renyi = np.log(np.maximum(renyi_core, eps_prob)) / (1.0 - alpha) / log_n
        h_renyi = np.clip(h_renyi, 0.0, 1.0)

    if mode == "renyi":
        return h_renyi
    if mode == "hybrid":
        blend = float(np.clip(renyi_entropy_blend, 0.0, 1.0))
        return (1.0 - blend) * h_shannon + blend * h_renyi

    return h_shannon


def support_entropy_batch_torch(
    pi,
    eps_prob: float = 1e-12,
    mode: str = "shannon",
    renyi_alpha: float = 2.0,
    renyi_entropy_blend: float = 0.5,
):
    n_hyp = int(pi.shape[1])
    log_n = np.log(max(n_hyp, 2))
    pi_safe = pi.clamp(min=eps_prob)
    pi_safe = pi_safe / pi_safe.sum(dim=1, keepdim=True).clamp(min=_EPS_PROB)
    h_shannon = -(pi_safe * torch.log(pi_safe)).sum(dim=1) / log_n

    mode = str(mode).strip().lower()
    if mode == "shannon":
        return h_shannon

    alpha = float(renyi_alpha)
    if alpha <= 0.0:
        alpha = 2.0
    if abs(alpha - 1.0) < 1e-6:
        h_renyi = h_shannon
    else:
        renyi_core = (pi_safe ** alpha).sum(dim=1).clamp(min=_EPS_PROB)
        h_renyi = torch.log(renyi_core) / (1.0 - alpha) / log_n
        h_renyi = h_renyi.clamp(min=0.0, max=1.0)

    if mode == "renyi":
        return h_renyi
    if mode == "hybrid":
        blend = float(np.clip(renyi_entropy_blend, 0.0, 1.0))
        return (1.0 - blend) * h_shannon + blend * h_renyi

    return h_shannon


def kl_divergence_batch_np(
    pi_post: NDArray[np.float64],
    pi_prior: NDArray[np.float64],
    eps_prob: float = 1e-12,
) -> NDArray[np.float64]:
    p = np.maximum(np.asarray(pi_post, dtype=np.float64), eps_prob)
    q = np.maximum(np.asarray(pi_prior, dtype=np.float64), eps_prob)
    p /= np.sum(p, axis=1, keepdims=True)
    q /= np.sum(q, axis=1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=1)


def kl_divergence_batch_torch(
    pi_post,
    pi_prior,
    eps_prob: float = 1e-12,
):
    p = pi_post.clamp(min=eps_prob)
    q = pi_prior.clamp(min=eps_prob)
    p = p / p.sum(dim=1, keepdim=True).clamp(min=_EPS_PROB)
    q = q / q.sum(dim=1, keepdim=True).clamp(min=_EPS_PROB)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=1)


def covariance_trace_proxy_actions_np(
    pi: NDArray[np.float64],                      
    mu: NDArray[np.float64],                      
    P: NDArray[np.float64],                       
    g_norm_matrix: NDArray[np.float64],                  
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    amp_sq = np.maximum(mu[:, 1:] ** 2 + P[:, 1:], 0.0)                  
    lam = amp_sq[:, None, :] * g_norm_matrix[None, :, :]                     
    w = np.maximum(pi[:, 1:], 0.0)
    w /= np.maximum(np.sum(w, axis=1, keepdims=True), eps)
    mean = np.sum(w[:, None, :] * lam, axis=2)          
    var = np.sum(w[:, None, :] * (lam - mean[:, :, None]) ** 2, axis=2)
    cv2 = var / np.maximum(mean ** 2, eps)
    return np.clip(cv2 / (1.0 + cv2), 0.0, 1.0)


def covariance_trace_proxy_row_np(
    pi: NDArray[np.float64],                      
    mu: NDArray[np.float64],                      
    P: NDArray[np.float64],                       
    g_norm_row: NDArray[np.float64],              
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    amp_sq = np.maximum(mu[:, 1:] ** 2 + P[:, 1:], 0.0)
    lam = amp_sq * np.maximum(g_norm_row[:, 1:], 0.0)
    w = np.maximum(pi[:, 1:], 0.0)
    w /= np.maximum(np.sum(w, axis=1, keepdims=True), eps)
    mean = np.sum(w * lam, axis=1)
    var = np.sum(w * (lam - mean[:, None]) ** 2, axis=1)
    cv2 = var / np.maximum(mean ** 2, eps)
    return np.clip(cv2 / (1.0 + cv2), 0.0, 1.0)


def covariance_trace_proxy_actions_torch(
    pi,
    mu,
    P,
    g_norm_matrix,
    eps: float = 1e-12,
):
    amp_sq = (mu[:, 1:] ** 2 + P[:, 1:]).clamp(min=0.0)
    lam = amp_sq.unsqueeze(1) * g_norm_matrix.unsqueeze(0)
    w = pi[:, 1:].clamp(min=0.0)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=max(eps, _EPS_PROB))
    mean = (w.unsqueeze(1) * lam).sum(dim=2)
    var = (w.unsqueeze(1) * (lam - mean.unsqueeze(2)) ** 2).sum(dim=2)
    cv2 = var / mean.pow(2).clamp(min=max(eps, _EPS_PROB))
    return (cv2 / (1.0 + cv2)).clamp(min=0.0, max=1.0)


def covariance_trace_proxy_row_torch(
    pi,
    mu,
    P,
    g_norm_row,
    eps: float = 1e-12,
):
    amp_sq = (mu[:, 1:] ** 2 + P[:, 1:]).clamp(min=0.0)
    lam = amp_sq * g_norm_row[:, 1:].clamp(min=0.0)
    w = pi[:, 1:].clamp(min=0.0)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=max(eps, _EPS_PROB))
    mean = (w * lam).sum(dim=1)
    var = (w * (lam - mean.unsqueeze(1)) ** 2).sum(dim=1)
    cv2 = var / mean.pow(2).clamp(min=max(eps, _EPS_PROB))
    return (cv2 / (1.0 + cv2)).clamp(min=0.0, max=1.0)


                                                                        
                     
                                                                        

def batch_detectability_np(
    pi: NDArray[np.float64],                      
    mu: NDArray[np.float64],                      
    P: NDArray[np.float64],                       
    g_norm_matrix: NDArray[np.float64],                  
    eps_d: float = 1e-12,
) -> NDArray[np.float64]:
    B = pi.shape[0]
    L = g_norm_matrix.shape[0]

                                                                           
    weights = pi[:, 1:] * (mu[:, 1:] ** 2 + P[:, 1:])                  
    np.nan_to_num(weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                                                               
    d_raw = weights @ g_norm_matrix.T          
    np.maximum(d_raw, 0.0, out=d_raw)

                          
    d_max = np.max(d_raw, axis=1, keepdims=True)          
    d_norm = d_raw / (d_max + eps_d)
    np.nan_to_num(d_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return d_norm          


                                                                        
                                                   
                                                                        

def _confidence_weights_np(
    pi_nonnull: NDArray[np.float64],
    confidence_beta: float,
    eps: float = _EPS_PROB,
) -> NDArray[np.float64]:
    pi_nonnull = np.asarray(pi_nonnull, dtype=np.float64)
    if pi_nonnull.size == 0 or confidence_beta <= 0.0:
        return np.ones_like(pi_nonnull, dtype=np.float64)

    n_nonnull = int(pi_nonnull.shape[-1])
    if n_nonnull <= 1:
        return np.ones_like(pi_nonnull, dtype=np.float64)

    mass = np.sum(pi_nonnull, axis=-1, keepdims=True)
    mass = np.maximum(mass, eps)
    pi_norm = pi_nonnull / mass

    uniform = 1.0 / float(n_nonnull)
    denom = max(1.0 - uniform, eps)
    confidence = np.clip((pi_norm - uniform) / denom, 0.0, 1.0)
    return 1.0 + float(confidence_beta) * confidence


def _confidence_weights_torch(
    pi_nonnull,
    confidence_beta: float,
    eps: float = _EPS_PROB,
):
    if pi_nonnull.numel() == 0 or confidence_beta <= 0.0:
        return torch.ones_like(pi_nonnull, dtype=torch.float64)

    n_nonnull = int(pi_nonnull.shape[-1])
    if n_nonnull <= 1:
        return torch.ones_like(pi_nonnull, dtype=torch.float64)

    mass = pi_nonnull.sum(dim=-1, keepdim=True).clamp(min=eps)
    pi_norm = pi_nonnull / mass
    uniform = 1.0 / float(n_nonnull)
    denom = max(1.0 - uniform, eps)
    confidence = ((pi_norm - uniform) / denom).clamp(min=0.0, max=1.0)
    return 1.0 + float(confidence_beta) * confidence

def batch_detectability_prob_np(
    pi: NDArray[np.float64],                         
    mu: NDArray[np.float64],                         
    P: NDArray[np.float64],                          
    g_norm_matrix: NDArray[np.float64],                  
    gamma_th: float,
    steepness: float = 4.0,
    confidence_beta: float = 0.0,
) -> NDArray[np.float64]:
    pi = np.nan_to_num(np.asarray(pi, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(np.asarray(mu, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    P = np.nan_to_num(np.asarray(P, dtype=np.float64), nan=0.0, posinf=1e12, neginf=0.0)
    g_norm_matrix = np.nan_to_num(
        np.asarray(g_norm_matrix, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.maximum(g_norm_matrix, 0.0, out=g_norm_matrix)
    pi = np.maximum(pi, 0.0)
    pi_sum = np.sum(pi, axis=1, keepdims=True)
    pi = np.divide(
        pi,
        np.maximum(pi_sum, _EPS_PROB),
        out=np.full_like(pi, 1.0 / max(pi.shape[1], 1)),
        where=pi_sum > _EPS_PROB,
    )
    P = np.maximum(P, _EPS_VAR)
    if P.shape[1] > 0:
        P[:, 0] = 0.0

    n_nonnull = min(
        g_norm_matrix.shape[1],
        max(min(pi.shape[1], mu.shape[1], P.shape[1]) - 1, 0),
    )
    B = pi.shape[0]
    L = g_norm_matrix.shape[0]

    if n_nonnull == 0:
        return np.zeros((B, L), dtype=np.float64)

    g_norm_matrix = g_norm_matrix[:, :n_nonnull]
    amp_sq = mu[:, 1:1 + n_nonnull] ** 2 + P[:, 1:1 + n_nonnull]           
    amp_sq = np.nan_to_num(amp_sq, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(amp_sq, 0.0, out=amp_sq)
    ncp = g_norm_matrix[None, :, :] * amp_sq[:, None, :]                        

    x = steepness * (ncp / max(gamma_th, 1e-30) - 1.0)
    p_det = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))                         
    p_det = np.nan_to_num(p_det, nan=0.0, posinf=1.0, neginf=0.0)

    pi_nonnull = pi[:, 1:1 + n_nonnull]
    conf_weights = _confidence_weights_np(pi_nonnull, confidence_beta)
    weighted_pi = pi_nonnull * conf_weights
    d_prob = (p_det * weighted_pi[:, None, :]).sum(axis=2)                   
    d_prob = np.nan_to_num(d_prob, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(d_prob, 0.0, 1.0)


def batch_detectability_prob_torch(
    pi,                         
    mu,                         
    P,                          
    g_norm_matrix,                  
    gamma_th: float,
    steepness: float = 4.0,
    confidence_beta: float = 0.0,
):
    n_nonnull = g_norm_matrix.shape[1]
    if n_nonnull == 0:
        return torch.zeros(pi.shape[0], g_norm_matrix.shape[0],
                           dtype=torch.float64, device=pi.device)

    pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    P = torch.nan_to_num(P, nan=0.0, posinf=1e12, neginf=0.0).clamp(min=_EPS_VAR)
    if P.shape[1] > 0:
        P[:, 0] = 0.0
    pi = pi / pi.sum(dim=1, keepdim=True).clamp(min=_EPS_PROB)

    amp_sq = mu[:, 1:1 + n_nonnull] ** 2 + P[:, 1:1 + n_nonnull]
    amp_sq = torch.nan_to_num(amp_sq, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    ncp = g_norm_matrix.unsqueeze(0) * amp_sq.unsqueeze(1)              

    x = steepness * (ncp / max(gamma_th, 1e-30) - 1.0)
    p_det = torch.sigmoid(x)              
    p_det = torch.nan_to_num(p_det, nan=0.0, posinf=1.0, neginf=0.0)

    pi_nonnull = pi[:, 1:1 + n_nonnull]
    conf_weights = _confidence_weights_torch(pi_nonnull, confidence_beta)
    weighted_pi = pi_nonnull * conf_weights
    d_prob = (p_det * weighted_pi[:, None, :]).sum(dim=2)          
    return torch.nan_to_num(d_prob, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1.0)


                                                                        
                                   
                                                                        

def vectorized_horizon2_plan(
    belief_pi: NDArray[np.float64],             
    belief_mu: NDArray[np.float64],             
    belief_P: NDArray[np.float64],              
    G_all: NDArray[np.float64],                       
    g_norm_sq_all: NDArray[np.float64],              
    g_norm_matrix: NDArray[np.float64],                  
    act_costs: NDArray[np.float64],         
    n_scenarios: int,
    rng: np.random.Generator,
    q_c: float,
    rho_u: float,
    xi_weight: float,
    lambda_H: float,
    P0: float,
    support_entropy_mode: str = "shannon",
    renyi_alpha: float = 2.0,
    renyi_entropy_blend: float = 0.5,
    eps_d: float = 1e-12,
    eps_prob: float = 1e-12,
    gamma_th: float = 45.0,
    steepness: float = 4.0,
    exploit_confidence_beta: float = 0.0,
    terminal_xi_bonus: float = 1.0,
    terminal_entropy_bonus: float = 0.0,
    terminal_margin_bonus: float = 0.0,
    entropy_drop_bonus: float = 0.0,
    cov_trace_enable: bool = False,
    cov_trace_weight: float = 0.0,
    cov_trace_terminal_bonus: float = 0.0,
    info_gain_bonus: float = 0.0,
    step1_xi_coupling: float = 0.0,
    step1_exploit_coupling: float = 0.0,
    step1_dual_shared_coupling: float = 0.0,
    step1_phase_dual_coupling: float = 0.0,
    step1_temporal_coupling: float = 0.0,
    step1_temporal_detect_ref: float = 0.55,
    step1_temporal_detect_floor: float = 0.0,
    topk_ratio: float = 0.20,
    topk_min: int = 2,
    topk_screen_scenarios: int = 1,
    exploit_uncertainty_gate: bool = False,
    exploit_gate_strength: float = 0.0,
    exploit_gate_floor: float = 0.30,
) -> Tuple[int, List[float]]:
                                                                            
    G_all = np.nan_to_num(
        np.asarray(G_all, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    g_norm_sq_all = np.nan_to_num(
        np.asarray(g_norm_sq_all, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.maximum(g_norm_sq_all, 0.0, out=g_norm_sq_all)
    g_norm_matrix = np.nan_to_num(
        np.asarray(g_norm_matrix, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.maximum(g_norm_matrix, 0.0, out=g_norm_matrix)
    act_costs = np.nan_to_num(
        np.asarray(act_costs, dtype=np.float64),
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )
    np.maximum(act_costs, 0.0, out=act_costs)

    L, n_hyp, nu = G_all.shape

    _check_torch()                                         
    use_gpu = _USE_TORCH and L * n_scenarios > 50                                             

    if use_gpu:
        return _vectorized_h2_torch(
            belief_pi, belief_mu, belief_P,
            G_all, g_norm_sq_all, g_norm_matrix, act_costs,
            n_scenarios, rng, q_c, rho_u, xi_weight, lambda_H, P0,
            support_entropy_mode, renyi_alpha, renyi_entropy_blend,
            eps_d, eps_prob, gamma_th, steepness, exploit_confidence_beta,
            terminal_xi_bonus,
            terminal_entropy_bonus,
            terminal_margin_bonus,
            entropy_drop_bonus,
            cov_trace_enable,
            cov_trace_weight,
            cov_trace_terminal_bonus,
            info_gain_bonus,
            step1_xi_coupling,
            step1_exploit_coupling,
            step1_dual_shared_coupling,
            step1_phase_dual_coupling,
            step1_temporal_coupling, step1_temporal_detect_ref,
            step1_temporal_detect_floor,
            topk_ratio, topk_min, topk_screen_scenarios,
            exploit_uncertainty_gate, exploit_gate_strength, exploit_gate_floor,
        )
    else:
        return _vectorized_h2_numpy(
            belief_pi, belief_mu, belief_P,
            G_all, g_norm_sq_all, g_norm_matrix, act_costs,
            n_scenarios, rng, q_c, rho_u, xi_weight, lambda_H, P0,
            support_entropy_mode, renyi_alpha, renyi_entropy_blend,
            eps_d, eps_prob, gamma_th, steepness, exploit_confidence_beta,
            terminal_xi_bonus,
            terminal_entropy_bonus,
            terminal_margin_bonus,
            entropy_drop_bonus,
            cov_trace_enable,
            cov_trace_weight,
            cov_trace_terminal_bonus,
            info_gain_bonus,
            step1_xi_coupling,
            step1_exploit_coupling,
            step1_dual_shared_coupling,
            step1_phase_dual_coupling,
            step1_temporal_coupling, step1_temporal_detect_ref,
            step1_temporal_detect_floor,
            topk_ratio, topk_min, topk_screen_scenarios,
            exploit_uncertainty_gate, exploit_gate_strength, exploit_gate_floor,
        )


def _vectorized_h2_numpy(
    belief_pi, belief_mu, belief_P,
    G_all, g_norm_sq_all, g_norm_matrix, act_costs,
    n_scenarios, rng, q_c, rho_u, xi_weight, lambda_H, P0,
    support_entropy_mode="shannon",
    renyi_alpha=2.0,
    renyi_entropy_blend=0.5,
    eps_d=1e-12,
    eps_prob=1e-12,
    gamma_th=45.0,
    steepness=4.0,
    exploit_confidence_beta=0.0,
    terminal_xi_bonus=1.0,
    terminal_entropy_bonus=0.0,
    terminal_margin_bonus=0.0,
    entropy_drop_bonus=0.0,
    cov_trace_enable=False,
    cov_trace_weight=0.0,
    cov_trace_terminal_bonus=0.0,
    info_gain_bonus=0.0,
    step1_xi_coupling=0.0,
    step1_exploit_coupling=0.0,
    step1_dual_shared_coupling=0.0,
    step1_phase_dual_coupling=0.0,
    step1_temporal_coupling=0.0,
    step1_temporal_detect_ref=0.55,
    step1_temporal_detect_floor=0.0,
    topk_ratio=0.20, topk_min=2, topk_screen_scenarios=1,
    exploit_uncertainty_gate=False,
    exploit_gate_strength=0.0,
    exploit_gate_floor=0.30,
):
    L, n_hyp, nu = G_all.shape
    S = n_scenarios
    n_nonnull = g_norm_matrix.shape[1]

                                                           
    ratio = float(np.clip(topk_ratio, 0.0, 1.0))
    min_k = max(int(topk_min), 1)
    if ratio >= 0.999:
        K = L
    else:
        K = min(L, max(int(np.ceil(ratio * L)), min_k))
    use_topk = L > K + 2
    gate_on = bool(exploit_uncertainty_gate)
    gate_strength = max(float(exploit_gate_strength), 0.0)
    gate_floor = float(np.clip(exploit_gate_floor, 0.0, 1.0))
    terminal_margin_bonus = max(float(terminal_margin_bonus), 0.0)
    entropy_drop_bonus = max(float(entropy_drop_bonus), 0.0)
    cov_trace_enable = bool(cov_trace_enable)
    cov_trace_weight = max(float(cov_trace_weight), 0.0)
    cov_trace_terminal_bonus = max(float(cov_trace_terminal_bonus), 0.0)
    cov_trace_active = cov_trace_enable and (
        cov_trace_weight > 0.0 or cov_trace_terminal_bonus > 0.0
    )
    info_gain_bonus = max(float(info_gain_bonus), 0.0)
    step1_xi_coupling = max(float(step1_xi_coupling), 0.0)
    step1_exploit_coupling = max(float(step1_exploit_coupling), 0.0)
    step1_dual_shared_coupling = max(float(step1_dual_shared_coupling), 0.0)
    step1_phase_dual_coupling = max(float(step1_phase_dual_coupling), 0.0)
    step1_temporal_coupling = max(float(step1_temporal_coupling), 0.0)
    step1_temporal_detect_floor = float(np.clip(step1_temporal_detect_floor, 0.0, 1.0))
    step1_temporal_detect_ref = max(float(step1_temporal_detect_ref), 1e-6)
    if step1_temporal_detect_floor >= step1_temporal_detect_ref:
        step1_temporal_detect_ref = step1_temporal_detect_floor + 1e-6

                                                              
    pi = np.asarray(belief_pi, dtype=np.float64).copy()
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    pi = np.maximum(pi, 0.0)
    pi_sum = float(np.sum(pi))
    if pi_sum <= _EPS_PROB:
        pi = np.full(n_hyp, 1.0 / max(n_hyp, 1), dtype=np.float64)
    else:
        pi /= pi_sum

    mu = np.asarray(belief_mu, dtype=np.float64).copy()
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    P = np.asarray(belief_P, dtype=np.float64).copy()
    P = np.nan_to_num(P, nan=0.0, posinf=1e12, neginf=0.0)
    P = np.maximum(P, _EPS_VAR)
    if P.size > 0:
        P[0] = 0.0

                                                    
    h_samples = rng.choice(n_hyp, size=S, p=pi)
    c_samples = np.zeros(S, dtype=np.float64)
    nonnull = h_samples > 0
    if np.any(nonnull):
        h_nonnull = h_samples[nonnull]
        mu_sel = mu[h_nonnull]
        std_sel = np.sqrt(np.maximum(P[h_nonnull], _EPS_VAR))
        c_samples[nonnull] = rng.normal(loc=mu_sel, scale=std_sel)

                                                             
    amp_sq_prior = mu[1:1 + n_nonnull] ** 2 + P[1:1 + n_nonnull]
    amp_sq_prior = np.nan_to_num(amp_sq_prior, nan=0.0, posinf=0.0, neginf=0.0)
    amp_sq_prior = np.maximum(amp_sq_prior, 0.0)
    ncp_prior = g_norm_matrix * amp_sq_prior[None, :]           
    x_prior = steepness * (ncp_prior / max(gamma_th, 1e-30) - 1.0)
    p_det_prior = 1.0 / (1.0 + np.exp(-np.clip(x_prior, -50, 50)))           
    p_det_prior = np.nan_to_num(p_det_prior, nan=0.0, posinf=1.0, neginf=0.0)
    pi_nonnull = np.nan_to_num(
        pi[1:1 + n_nonnull], nan=0.0, posinf=0.0, neginf=0.0
    )
    conf_weights = _confidence_weights_np(
        pi_nonnull[None, :],
        float(exploit_confidence_beta),
    ).reshape(-1)
    weighted_pi_nonnull = pi_nonnull * conf_weights
    d_prob_vec = np.sum(p_det_prior * weighted_pi_nonnull[None, :], axis=1)        
    d_prob_vec = np.nan_to_num(d_prob_vec, nan=0.0, posinf=0.0, neginf=0.0)
    d_prob_vec = np.maximum(d_prob_vec, 0.0)
    if gate_on and gate_strength > 0.0:
        h_supp = float(
            support_entropy_batch_np(
                pi[None, :],
                eps_prob=eps_prob,
                mode=support_entropy_mode,
                renyi_alpha=renyi_alpha,
                renyi_entropy_blend=renyi_entropy_blend,
            )[0]
        )
        exploit_w = float(np.clip(1.0 - gate_strength * h_supp, gate_floor, 1.0))
        d_prob_vec = d_prob_vec * exploit_w

                                                                      
                                                                     
                                                                      
    if use_topk:
        n_screen = min(S, max(int(topk_screen_scenarios), 1))
        screen_cost = np.zeros(L, dtype=np.float64)
        rng_screen = np.random.default_rng(rng.integers(0, 2**63))

        for s_idx in range(n_screen):
            h0 = int(h_samples[s_idx])
            c0 = float(c_samples[s_idx])
            eps_screen = rng_screen.standard_normal((L, nu))
            if h0 > 0:
                y_screen = G_all[:, h0, :] * c0 + eps_screen
            else:
                y_screen = eps_screen

            pi_s = np.broadcast_to(pi[None, :], (L, n_hyp)).copy()
            mu_s = np.broadcast_to(mu[None, :], (L, n_hyp)).copy()
            P_s = np.broadcast_to(P[None, :], (L, n_hyp)).copy()
            pi1_s, mu1_s, P1_s = batch_belief_update_np(
                pi_s, mu_s, P_s, y_screen, G_all, g_norm_sq_all, q_c,
            )
            xi_s = batch_uncertainty_np(
                pi1_s,
                P1_s,
                lambda_H,
                P0,
                eps_prob,
                support_entropy_mode,
                renyi_alpha,
                renyi_entropy_blend,
            )
            if cov_trace_active:
                cov_s = covariance_trace_proxy_actions_np(
                    pi1_s, mu1_s, P1_s, g_norm_matrix, eps=eps_prob
                )          
                                                                 
                cov_diag = np.diag(cov_s)
            else:
                cov_diag = 0.0
            step_cost = (
                -d_prob_vec
                + xi_weight * (xi_s + cov_trace_weight * cov_diag)
                + rho_u * act_costs
            )
            screen_cost += np.nan_to_num(
                step_cost, nan=np.inf, posinf=np.inf, neginf=np.inf
            )

        screen_cost /= float(n_screen)
        topk = np.argsort(screen_cost)[:K]
    else:
        topk = np.arange(L)
        K = L

                                                                      
                                                                      
                                                                      
    KL = K * L

    G_step1 = np.tile(G_all, (K, 1, 1))                           
    gns_step1 = np.tile(g_norm_sq_all, (K, 1))                 

    G_topk = G_all[topk]                                           
    gns_topk = g_norm_sq_all[topk]                             
    d_prob_topk = d_prob_vec[topk]                       
    act_costs_topk = act_costs[topk]                     

    pair_costs = np.zeros((K, L), dtype=np.float64)

    for s in range(S):
        h_s = int(h_samples[s])
        c_s = float(c_samples[s])

                                                                      
        eps0 = rng.standard_normal((K, nu))
        if h_s > 0:
            y0 = G_topk[:, h_s, :] * c_s + eps0
        else:
            y0 = eps0

        pi0 = np.broadcast_to(pi[None, :], (K, n_hyp)).copy()
        mu0 = np.broadcast_to(mu[None, :], (K, n_hyp)).copy()
        P0_arr = np.broadcast_to(P[None, :], (K, n_hyp)).copy()

        pi1, mu1, P1 = batch_belief_update_np(
            pi0, mu0, P0_arr, y0, G_topk, gns_topk, q_c,
        )

        xi0 = batch_uncertainty_np(
            pi1,
            P1,
            lambda_H,
            P0,
            eps_prob,
            support_entropy_mode,
            renyi_alpha,
            renyi_entropy_blend,
        )
        if cov_trace_active:
            cov0_all = covariance_trace_proxy_actions_np(
                pi1, mu1, P1, g_norm_matrix, eps=eps_prob
            )          
            cov0 = np.clip(np.diag(cov0_all), 0.0, 1.0)
        else:
            cov0 = np.zeros(K, dtype=np.float64)
        if info_gain_bonus > 0.0:
            pi_prior_k = np.broadcast_to(pi[None, :], (K, n_hyp))
            kl0 = kl_divergence_batch_np(pi1, pi_prior_k, eps_prob=eps_prob)
        else:
            kl0 = np.zeros(K, dtype=np.float64)
        cost0 = (
            -d_prob_topk
            + xi_weight * (xi0 + cov_trace_weight * cov0)
            + rho_u * act_costs_topk
            - xi_weight * info_gain_bonus * kl0
        )

        d_prob_1 = batch_detectability_prob_np(
            pi1, mu1, P1, g_norm_matrix, gamma_th, steepness,
            confidence_beta=float(exploit_confidence_beta),
        )          

                                                           
        pi1_exp = np.repeat(pi1, L, axis=0)
        mu1_exp = np.repeat(mu1, L, axis=0)
        P1_exp = np.repeat(P1, L, axis=0)

        eps1_all = rng.standard_normal((KL, nu))

        if h_s > 0:
            signal_j = G_all[:, h_s, :] * c_s
            signal_1 = np.tile(signal_j, (K, 1))
            y1_all = signal_1 + eps1_all
        else:
            y1_all = eps1_all

        pi2, mu2, P2 = batch_belief_update_np(
            pi1_exp, mu1_exp, P1_exp, y1_all,
            G_step1, gns_step1, q_c,
        )
        if info_gain_bonus > 0.0:
            kl1_all = kl_divergence_batch_np(pi2, pi1_exp, eps_prob=eps_prob)
            kl1_mat = kl1_all.reshape(K, L)
        else:
            kl1_mat = np.zeros((K, L), dtype=np.float64)

        xi1_all = batch_uncertainty_np(
            pi2,
            P2,
            lambda_H,
            P0,
            eps_prob,
            support_entropy_mode,
            renyi_alpha,
            renyi_entropy_blend,
        )
        xi1_mat = xi1_all.reshape(K, L)
        if cov_trace_active:
            cov1_all = covariance_trace_proxy_row_np(
                pi2, mu2, P2, gns_step1, eps=eps_prob
            )
            cov1_mat = cov1_all.reshape(K, L)
        else:
            cov1_mat = np.zeros((K, L), dtype=np.float64)

        h_supp0 = support_entropy_batch_np(
            pi1,
            eps_prob=eps_prob,
            mode=support_entropy_mode,
            renyi_alpha=renyi_alpha,
            renyi_entropy_blend=renyi_entropy_blend,
        )

        h_supp1_all = support_entropy_batch_np(
            pi2,
            eps_prob=eps_prob,
            mode=support_entropy_mode,
            renyi_alpha=renyi_alpha,
            renyi_entropy_blend=renyi_entropy_blend,
        )
        h_supp1_mat = h_supp1_all.reshape(K, L)
        if terminal_margin_bonus > 0.0 and n_hyp >= 2:
            top2 = np.partition(pi2, kth=n_hyp - 2, axis=1)[:, -2:]
            margin_all = np.maximum(top2[:, 1] - top2[:, 0], 0.0)
            terminal_margin_mat = margin_all.reshape(K, L)
        else:
            terminal_margin_mat = np.zeros((K, L), dtype=np.float64)
        if gate_on and gate_strength > 0.0:
            exploit_w_step1 = np.clip(
                1.0 - gate_strength * h_supp0, gate_floor, 1.0
            )[:, None]
            d_prob_1 = d_prob_1 * exploit_w_step1
        if step1_exploit_coupling > 0.0:
            exploit_w_coupled = np.clip(
                1.0 - step1_exploit_coupling * h_supp0,
                gate_floor,
                1.0,
            )[:, None]
            d_prob_1 = d_prob_1 * exploit_w_coupled

        entropy_drop_mat = np.maximum(h_supp0[:, None] - h_supp1_mat, 0.0)
        xi_inner = (
            terminal_xi_bonus * xi1_mat
            + terminal_entropy_bonus * h_supp1_mat
            - terminal_margin_bonus * terminal_margin_mat
            - entropy_drop_bonus * entropy_drop_mat
            - info_gain_bonus * kl1_mat
            + cov_trace_weight * cov1_mat
            + cov_trace_terminal_bonus * cov1_mat
        )
        if step1_xi_coupling > 0.0:
            xi_stage1_w = xi_weight * (1.0 + step1_xi_coupling * h_supp0)[:, None]
        else:
            xi_stage1_w = np.full((K, 1), xi_weight, dtype=np.float64)
        if step1_dual_shared_coupling > 0.0:
            shared_gate = np.clip(
                1.0 - step1_dual_shared_coupling * h_supp0,
                gate_floor,
                1.0,
            )[:, None]
            d_prob_1 = d_prob_1 * shared_gate
            xi_stage1_w = xi_stage1_w * (2.0 - shared_gate)
        if step1_phase_dual_coupling > 0.0:
                                                               
                                               
            phase_signal = h_supp0 * np.maximum(1.0 - d_prob_topk, 0.0)
            phase_gate = gate_floor + (1.0 - gate_floor) * np.exp(
                -step1_phase_dual_coupling * np.maximum(phase_signal, 0.0)
            )[:, None]
            phase_gate = np.clip(phase_gate, gate_floor, 1.0)
            d_prob_1 = d_prob_1 * phase_gate
            xi_stage1_w = xi_stage1_w * (2.0 - phase_gate) * 1.15
        if step1_temporal_coupling > 0.0:
            temporal_conf = np.maximum(d_prob_topk - step1_temporal_detect_floor, 0.0)
            temporal_norm = temporal_conf / (
                step1_temporal_detect_ref - step1_temporal_detect_floor
            )
            temporal_phase = np.clip(
                step1_temporal_coupling * temporal_norm,
                0.0,
                0.9,
            )[:, None]
            d_prob_1 = d_prob_1 * (1.0 - temporal_phase)
            xi_stage1_w = xi_stage1_w * (1.0 + temporal_phase)
        step1_cost = -d_prob_1 + xi_stage1_w * xi_inner + rho_u * act_costs[None, :]
        total = cost0[:, None] + step1_cost
        pair_costs += total / S

                            
    marginal_topk = np.min(pair_costs, axis=1)
    best_k = int(np.argmin(marginal_topk))
    best_first = int(topk[best_k])

    full_marginal = np.full(L, float(np.max(marginal_topk) + 1.0), dtype=np.float64)
    full_marginal[topk] = marginal_topk

    return best_first, full_marginal.tolist()


def _vectorized_h2_torch(
    belief_pi, belief_mu, belief_P,
    G_all_np, g_norm_sq_all_np, g_norm_matrix_np, act_costs_np,
    n_scenarios, rng, q_c, rho_u, xi_weight, lambda_H, P0,
    support_entropy_mode="shannon",
    renyi_alpha=2.0,
    renyi_entropy_blend=0.5,
    eps_d=1e-12,
    eps_prob=1e-12,
    gamma_th=45.0,
    steepness=4.0,
    exploit_confidence_beta=0.0,
    terminal_xi_bonus=1.0,
    terminal_entropy_bonus=0.0,
    terminal_margin_bonus=0.0,
    entropy_drop_bonus=0.0,
    cov_trace_enable=False,
    cov_trace_weight=0.0,
    cov_trace_terminal_bonus=0.0,
    info_gain_bonus=0.0,
    step1_xi_coupling=0.0,
    step1_exploit_coupling=0.0,
    step1_dual_shared_coupling=0.0,
    step1_phase_dual_coupling=0.0,
    step1_temporal_coupling=0.0,
    step1_temporal_detect_ref=0.55,
    step1_temporal_detect_floor=0.0,
    topk_ratio=0.20, topk_min=2, topk_screen_scenarios=1,
    exploit_uncertainty_gate=False,
    exploit_gate_strength=0.0,
    exploit_gate_floor=0.30,
):
    L, n_hyp, nu = G_all_np.shape
    S = n_scenarios
    LL = L * L
    gate_on = bool(exploit_uncertainty_gate)
    gate_strength = max(float(exploit_gate_strength), 0.0)
    gate_floor = float(np.clip(exploit_gate_floor, 0.0, 1.0))
    terminal_margin_bonus = max(float(terminal_margin_bonus), 0.0)
    entropy_drop_bonus = max(float(entropy_drop_bonus), 0.0)
    cov_trace_enable = bool(cov_trace_enable)
    cov_trace_weight = max(float(cov_trace_weight), 0.0)
    cov_trace_terminal_bonus = max(float(cov_trace_terminal_bonus), 0.0)
    cov_trace_active = cov_trace_enable and (
        cov_trace_weight > 0.0 or cov_trace_terminal_bonus > 0.0
    )
    info_gain_bonus = max(float(info_gain_bonus), 0.0)
    step1_xi_coupling = max(float(step1_xi_coupling), 0.0)
    step1_exploit_coupling = max(float(step1_exploit_coupling), 0.0)
    step1_dual_shared_coupling = max(float(step1_dual_shared_coupling), 0.0)
    step1_phase_dual_coupling = max(float(step1_phase_dual_coupling), 0.0)
    step1_temporal_coupling = max(float(step1_temporal_coupling), 0.0)
    step1_temporal_detect_floor = float(np.clip(step1_temporal_detect_floor, 0.0, 1.0))
    step1_temporal_detect_ref = max(float(step1_temporal_detect_ref), 1e-6)
    if step1_temporal_detect_floor >= step1_temporal_detect_ref:
        step1_temporal_detect_ref = step1_temporal_detect_floor + 1e-6

                         
    G_all = _to_tensor(G_all_np)
    g_norm_sq_all = _to_tensor(g_norm_sq_all_np)
    g_norm_matrix = _to_tensor(g_norm_matrix_np)
    act_costs = _to_tensor(act_costs_np)

    b_pi = _to_tensor(belief_pi)
    b_mu = _to_tensor(belief_mu)
    b_P = _to_tensor(belief_P)

                                 
    pi_np = np.asarray(belief_pi, dtype=np.float64).copy()
    pi_np = np.nan_to_num(pi_np, nan=0.0, posinf=0.0, neginf=0.0)
    pi_np = np.maximum(pi_np, 0.0)
    pi_sum = float(np.sum(pi_np))
    if pi_sum <= _EPS_PROB:
        pi_np = np.full(n_hyp, 1.0 / max(n_hyp, 1), dtype=np.float64)
    else:
        pi_np = pi_np / pi_sum
    h_samples = rng.choice(n_hyp, size=S, p=pi_np)
    c_samples = np.zeros(S, dtype=np.float64)
    nonnull = h_samples > 0
    if np.any(nonnull):
        h_nonnull = h_samples[nonnull]
        mu_sel = np.asarray(belief_mu, dtype=np.float64)[h_nonnull]
        P_sel = np.asarray(belief_P, dtype=np.float64)[h_nonnull]
        std_sel = np.sqrt(np.maximum(np.abs(P_sel), 1e-30))
        c_samples[nonnull] = rng.normal(loc=mu_sel, scale=std_sel)

    pair_costs = torch.zeros((L, L), dtype=torch.float64, device=_DEVICE)

                                               
    n_nonnull = g_norm_matrix.shape[1]
    amp_sq_prior = b_mu[1:1 + n_nonnull] ** 2 + b_P[1:1 + n_nonnull]
    ncp_prior = g_norm_matrix * amp_sq_prior.unsqueeze(0)           
    x_prior = steepness * (ncp_prior / max(gamma_th, 1e-30) - 1.0)
    p_det_prior = torch.sigmoid(x_prior)           
    pi_nonnull = b_pi[1:1 + n_nonnull]
    conf_weights = _confidence_weights_torch(
        pi_nonnull.unsqueeze(0),
        float(exploit_confidence_beta),
    ).squeeze(0)
    weighted_pi_nonnull = pi_nonnull * conf_weights
    d_prob_prior = p_det_prior @ weighted_pi_nonnull        
    if gate_on and gate_strength > 0.0:
        h_supp = support_entropy_batch_torch(
            b_pi.unsqueeze(0),
            eps_prob=eps_prob,
            mode=support_entropy_mode,
            renyi_alpha=renyi_alpha,
            renyi_entropy_blend=renyi_entropy_blend,
        )[0]
        exploit_w = torch.clamp(
            1.0 - gate_strength * h_supp, min=gate_floor, max=1.0
        )
        d_prob_prior = d_prob_prior * exploit_w

                                                         
    G_step1 = G_all.repeat(L, 1, 1)                           
    gns_step1 = g_norm_sq_all.repeat(L, 1)                

    for s in range(S):
        h_s = int(h_samples[s])
        c_s = float(c_samples[s])

                                                                      
        eps0 = _to_tensor(rng.standard_normal((L, nu)))
        if h_s > 0:
            y0 = G_all[:, h_s, :] * c_s + eps0
        else:
            y0 = eps0

        pi0 = b_pi.unsqueeze(0).expand(L, -1).clone()
        mu0 = b_mu.unsqueeze(0).expand(L, -1).clone()
        P0_arr = b_P.unsqueeze(0).expand(L, -1).clone()

        pi1, mu1, P1 = batch_belief_update_torch(
            pi0, mu0, P0_arr, y0,
            G_all, g_norm_sq_all, q_c,
        )

                                                                      
        xi0 = batch_uncertainty_torch(
            pi1,
            P1,
            lambda_H,
            P0,
            eps_prob,
            support_entropy_mode,
            renyi_alpha,
            renyi_entropy_blend,
        )
        if cov_trace_active:
            cov0_all = covariance_trace_proxy_actions_torch(
                pi1, mu1, P1, g_norm_matrix, eps=eps_prob
            )
            cov0 = torch.diagonal(cov0_all, offset=0, dim1=0, dim2=1)
        else:
            cov0 = torch.zeros(L, dtype=torch.float64, device=_DEVICE)
        if info_gain_bonus > 0.0:
            pi_prior_l = b_pi.unsqueeze(0).expand(L, -1)
            kl0 = kl_divergence_batch_torch(
                pi1, pi_prior_l, eps_prob=eps_prob
            )
        else:
            kl0 = torch.zeros(L, dtype=torch.float64, device=_DEVICE)
        cost0 = (
            -d_prob_prior
            + xi_weight * (xi0 + cov_trace_weight * cov0)
            + rho_u * act_costs
            - xi_weight * info_gain_bonus * kl0
        )        

                                                          
        d_prob_1 = batch_detectability_prob_torch(
            pi1, mu1, P1, g_norm_matrix, gamma_th, steepness,
            confidence_beta=float(exploit_confidence_beta),
        )          
        h_supp0 = support_entropy_batch_torch(
            pi1,
            eps_prob=eps_prob,
            mode=support_entropy_mode,
            renyi_alpha=renyi_alpha,
            renyi_entropy_blend=renyi_entropy_blend,
        )
        if gate_on and gate_strength > 0.0:
            exploit_w_step1 = torch.clamp(
                1.0 - gate_strength * h_supp0, min=gate_floor, max=1.0
            ).unsqueeze(1)
            d_prob_1 = d_prob_1 * exploit_w_step1
        if step1_exploit_coupling > 0.0:
            exploit_w_coupled = torch.clamp(
                1.0 - step1_exploit_coupling * h_supp0,
                min=gate_floor,
                max=1.0,
            ).unsqueeze(1)
            d_prob_1 = d_prob_1 * exploit_w_coupled

                                                              
        pi1_exp = pi1.repeat_interleave(L, dim=0)                
        mu1_exp = mu1.repeat_interleave(L, dim=0)
        P1_exp = P1.repeat_interleave(L, dim=0)

        eps1 = _to_tensor(rng.standard_normal((LL, nu)))
        if h_s > 0:
            signal_j = G_all[:, h_s, :] * c_s                  
            signal_1 = signal_j.repeat(L, 1)                     
            y1_all = signal_1 + eps1
        else:
            y1_all = eps1

        pi2, mu2, P2 = batch_belief_update_torch(
            pi1_exp, mu1_exp, P1_exp, y1_all,
            G_step1, gns_step1, q_c,
        )
        if info_gain_bonus > 0.0:
            kl1_all = kl_divergence_batch_torch(
                pi2, pi1_exp, eps_prob=eps_prob
            )
            kl1_mat = kl1_all.reshape(L, L)
        else:
            kl1_mat = torch.zeros((L, L), dtype=torch.float64, device=_DEVICE)

        xi1_all = batch_uncertainty_torch(
            pi2,
            P2,
            lambda_H,
            P0,
            eps_prob,
            support_entropy_mode,
            renyi_alpha,
            renyi_entropy_blend,
        )         
        xi1_mat = xi1_all.reshape(L, L)
        if cov_trace_active:
            cov1_all = covariance_trace_proxy_row_torch(
                pi2, mu2, P2, gns_step1, eps=eps_prob
            )
            cov1_mat = cov1_all.reshape(L, L)
        else:
            cov1_mat = torch.zeros((L, L), dtype=torch.float64, device=_DEVICE)

        h_supp1_all = support_entropy_batch_torch(
            pi2,
            eps_prob=eps_prob,
            mode=support_entropy_mode,
            renyi_alpha=renyi_alpha,
            renyi_entropy_blend=renyi_entropy_blend,
        )
        h_supp1_mat = h_supp1_all.reshape(L, L)
        if terminal_margin_bonus > 0.0 and n_hyp >= 2:
            top2 = torch.topk(pi2, k=2, dim=1).values
            terminal_margin_mat = torch.clamp(
                top2[:, 0] - top2[:, 1], min=0.0
            ).reshape(L, L)
        else:
            terminal_margin_mat = torch.zeros(
                (L, L), dtype=torch.float64, device=_DEVICE
            )
        entropy_drop_mat = (h_supp0.unsqueeze(1) - h_supp1_mat).clamp(min=0.0)

                                                                       
        xi_inner = (
            terminal_xi_bonus * xi1_mat
            + terminal_entropy_bonus * h_supp1_mat
            - terminal_margin_bonus * terminal_margin_mat
            - entropy_drop_bonus * entropy_drop_mat
            - info_gain_bonus * kl1_mat
            + cov_trace_weight * cov1_mat
            + cov_trace_terminal_bonus * cov1_mat
        )
        if step1_xi_coupling > 0.0:
            xi_stage1_w = xi_weight * (
                1.0 + step1_xi_coupling * h_supp0
            ).unsqueeze(1)
        else:
            xi_stage1_w = torch.full(
                (L, 1), float(xi_weight), dtype=torch.float64, device=_DEVICE
            )
        if step1_dual_shared_coupling > 0.0:
            shared_gate = torch.clamp(
                1.0 - step1_dual_shared_coupling * h_supp0,
                min=gate_floor,
                max=1.0,
            ).unsqueeze(1)
            d_prob_1 = d_prob_1 * shared_gate
            xi_stage1_w = xi_stage1_w * (2.0 - shared_gate)
        if step1_phase_dual_coupling > 0.0:
            phase_signal = h_supp0 * torch.clamp(
                1.0 - d_prob_prior, min=0.0, max=1.0
            )
            phase_gate = gate_floor + (1.0 - gate_floor) * torch.exp(
                -step1_phase_dual_coupling
                * torch.clamp(phase_signal, min=0.0)
            ).unsqueeze(1)
            phase_gate = torch.clamp(
                phase_gate,
                min=gate_floor,
                max=1.0,
            )
            d_prob_1 = d_prob_1 * phase_gate
            xi_stage1_w = xi_stage1_w * (2.0 - phase_gate) * 1.15
        if step1_temporal_coupling > 0.0:
            temporal_conf = torch.clamp(
                d_prob_prior - step1_temporal_detect_floor, min=0.0
            )
            temporal_norm = temporal_conf / (
                step1_temporal_detect_ref - step1_temporal_detect_floor
            )
            temporal_phase = torch.clamp(
                step1_temporal_coupling * temporal_norm,
                min=0.0,
                max=0.9,
            ).unsqueeze(1)
            d_prob_1 = d_prob_1 * (1.0 - temporal_phase)
            xi_stage1_w = xi_stage1_w * (1.0 + temporal_phase)
        step1_cost = -d_prob_1 + xi_stage1_w * xi_inner + rho_u * act_costs.unsqueeze(0)

        total = cost0.unsqueeze(1) + step1_cost          
        pair_costs += total / S

    marginal = pair_costs.min(dim=1).values
    best_first = int(marginal.argmin().item())
    return best_first, _to_numpy(marginal).tolist()
