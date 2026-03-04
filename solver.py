import numpy as np
import torch
from graph_builder import build_coefficient_graph

def build_hankel_matrix(a, r):
    n = len(a)
    max_row = n - r
    if max_row <= 0:
        return np.array([])
    H = np.zeros((max_row, r))
    for i in range(max_row):
        for j in range(r):
            H[i, j] = a[i + j]
    return H

def solve_recurrence_coefficients(H):
    if H.size == 0:
        return np.array([])
    try:
        m, n = H.shape
        if n < 2 or m < n - 1:
            return np.array([])
        X = H[:, :-1]
        y = H[:, -1]
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return c
    except np.linalg.LinAlgError:
        return np.array([])

def find_roots(c):
    r = len(c)
    if r == 0:
        return np.array([])
    # Reverse the coefficients to get the correct polynomial
    c_reversed = c[::-1]
    poly = np.concatenate([[1], -c_reversed])
    roots = np.roots(poly)
    return roots

def solve_vandermonde(roots, a):
    r = len(roots)
    n = len(a)
    # Use all available coefficients to solve for weights
    V = np.vander(roots, n, increasing=True)
    try:
        weights = np.linalg.lstsq(V.T, a, rcond=None)[0]
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(V.T, a, rcond=None)[0]
    return weights

def compute_residual(a, roots, weights):
    r = len(roots)
    n = len(a)
    max_residual = 0.0
    for i in range(n):
        predicted = np.sum(weights * (roots ** i))
        current_residual = abs(a[i] - predicted)
        if current_residual > max_residual:
            max_residual = current_residual
    return max_residual

def compute_residual_normalized_l2(a, roots, weights):
    n = len(a)
    # Predicted sequence
    predicted = np.sum(weights[:, None] * (roots[:, None] ** np.arange(n)), axis=0)
    norm_a = np.linalg.norm(a) + 1e-12
    eps_2 = np.linalg.norm(a - predicted) / norm_a
    return eps_2

def one_shot_reconstruction(a, r_hat, tau=1e-6, metric='eps_2'):
    tau = float(tau)
    H = build_hankel_matrix(a, r_hat + 1)
    c = solve_recurrence_coefficients(H)
    if len(c) != r_hat:
        return None
    roots = find_roots(c)
    if len(roots) != r_hat:
        return None
    weights = solve_vandermonde(roots, a)
    
    if metric == 'eps_max':
        residual = compute_residual(a, roots, weights)
    else:
        residual = compute_residual_normalized_l2(a, roots, weights)
        
    if residual < tau:
        return r_hat, roots, weights
    else:
        return None

def classical_sylvester(a, R_max, tau=1e-6, metric='eps_2'):
    tau = float(tau)
    d = len(a) - 1
    for r in range(1, R_max + 1):
        if d < 2 * r:
            continue
        H = build_hankel_matrix(a, r + 1)
        if H.size == 0 or H.shape[0] < r:
            continue
        c = solve_recurrence_coefficients(H)
        if len(c) != r:
            continue
        roots = find_roots(c)
        if len(roots) != r:
            continue
        weights = solve_vandermonde(roots, a)
        
        if metric == 'eps_max':
            residual = compute_residual(a, roots, weights)
        else:
            residual = compute_residual_normalized_l2(a, roots, weights)
            
        if residual < tau:
            return r, roots, weights
    return None

def svd_baseline(a, R_max, tau=1e-6, kappa=0.1, metric='eps_2'):
    tau = float(tau)
    n = len(a)
    max_hankel_size = min(R_max + 1, n // 2)
    H = build_hankel_matrix(a, max_hankel_size)
    if H.size == 0:
        return None
    try:
        U, S, Vt = np.linalg.svd(H)
        if len(S) == 0:
            return None
        threshold = kappa * S[0]
        r_hat = np.sum(S > threshold)
        r_hat = max(1, min(r_hat, R_max))
        result = one_shot_reconstruction(a, r_hat, tau, metric)
        if result is not None:
            return result
        else:
            for r_candidate in [r_hat - 1, r_hat + 1]:
                if 1 <= r_candidate <= R_max:
                    result = one_shot_reconstruction(a, r_candidate, tau, metric)
                    if result is not None:
                        return result
    except np.linalg.LinAlgError:
        pass
    return None

def gnn_predict_rank(gnn_model, graph):
    gnn_model.eval()
    device = next(gnn_model.parameters()).device
    with torch.no_grad():
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_weight = graph.edge_attr.to(device) if hasattr(graph, 'edge_attr') else None
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        rank_logits, _ = gnn_model(x, edge_index, batch, edge_weight)
        r_hat = torch.argmax(rank_logits, dim=1).item() + 1
    return r_hat

def hybrid_solver(a, d, gnn_model, R_max=10, base_tau=1e-6):
    base_tau = float(base_tau)
    graph = build_coefficient_graph(a, d, R_max)
    r_hat, stability_score = gnn_predict_rank_with_stability(gnn_model, graph)
    
    # Dynamic tau based on stability score (Log-linear interpolation)
    # High stability (1.0) -> strict tau (e.g. 1e-6)
    # Low stability (0.0) -> relaxed tau (e.g. 1e-2)
    
    log_tau_strict = np.log10(1e-4)  # Strict threshold for stable cases (relaxed from 1e-6)
    log_tau_loose = np.log10(5e-2)   # Loose threshold for unstable cases (relaxed from 1e-2)
    
    # stability 1 -> strict, stability 0 -> loose
    log_tau = log_tau_loose - stability_score * (log_tau_loose - log_tau_strict)
    tau = 10 ** log_tau
    
    # Ensure tau is at least the base_tau provided by user if strictness is preferred
    # or just use the computed tau as the paper suggests adaptive thresholding
    
    result = one_shot_reconstruction(a, r_hat, tau)
    if result is not None:
        return result
    else:
        for r_candidate in [r_hat - 1, r_hat + 1]:
            if 1 <= r_candidate <= R_max:
                result = one_shot_reconstruction(a, r_candidate, tau)
                if result is not None:
                    return result
    return None

def gnn_predict_rank_with_stability(gnn_model, graph):
    gnn_model.eval()
    device = next(gnn_model.parameters()).device
    with torch.no_grad():
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_weight = graph.edge_attr.to(device) if hasattr(graph, 'edge_attr') else None
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        rank_logits, stability_score = gnn_model(x, edge_index, batch, edge_weight)
        r_hat = torch.argmax(rank_logits, dim=1).item() + 1
        stability_score = stability_score.item()
    return r_hat, stability_score