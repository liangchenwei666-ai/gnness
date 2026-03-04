import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def extract_node_features(a_i, i, d, a):
    eps = 1e-12
    norm = np.linalg.norm(a) + eps
    normalized = a_i / norm
    position = i / d
    log_magnitude = np.log(1 + abs(normalized))
    
    # Return only the three features specified in the paper
    return [normalized, position, log_magnitude]

def deduplicate_and_weight_weighted(edges, weights, local_weight=1.0):
    edge_dict = {}
    
    # Process edges with weights (Hankel)
    for edge, weight in zip(edges, weights):
        if edge not in edge_dict:
            edge_dict[edge] = 0
        edge_dict[edge] += weight
        
    unique_edges = list(edge_dict.keys())
    final_weights = list(edge_dict.values())
    
    if final_weights:
        max_weight = max(final_weights)
        normalized_weights = [w / max_weight for w in final_weights]
    else:
        normalized_weights = []
        
    return unique_edges, final_weights, normalized_weights

def build_hankel_edges(d, R_max, delta=6):
    # Use a dictionary for edge multiplicity tracking
    edge_dict = {}
    # For a polynomial of degree d, there are d+1 coefficients
    n_coeff = d + 1
    
    # Limit the maximum window size to avoid excessive edges
    max_window_size = min(20, 2 * R_max + 1)
    
    for r in range(1, R_max + 1):
        Lr = min(2 * r + 1, max_window_size)
        # Skip if window size is too large for the current polynomial
        if Lr > n_coeff:
            continue
        
        for s in range(n_coeff - Lr + 1):
            window = list(range(s, s + Lr))
            # Generate edges based on constant index-sum (Hankel anti-diagonal)
            for idx1 in range(len(window)):
                for idx2 in range(idx1 + 1, len(window)): # Only consider upper triangle to avoid double counting
                    i = window[idx1]
                    j = window[idx2]
                    
                    # Condition from paper: i + j = 2s + t (implicitly satisfied by being in same window?)
                    # Actually, the paper says: "We connect two nodes i and j if they appear in the same Hankel window"
                    # AND "satisfy the Hankel structure constraint".
                    # But in a Hankel matrix H_r, the entry H[u, v] corresponds to a_{u+v}.
                    # The "Hankel window" usually refers to the coefficients involved in a specific sub-determinant or relation.
                    # A simpler interpretation widely used in such graphs:
                    # Connect i and j if they are part of the same linear recurrence relation window.
                    # The constraint |i-j| <= delta is a sparsity constraint.
                    
                    # Check sparsity constraint
                    if abs(i - j) <= delta:
                        edge = (i, j)
                        if edge not in edge_dict:
                            edge_dict[edge] = 0
                        edge_dict[edge] += 1
    
    # Convert dictionary to list of edges with multiplicities
    edges = []
    weights = []
    for edge, count in edge_dict.items():
        edges.append(edge)
        weights.append(count)
    
    # Generate reverse edges to make the graph undirected
    reverse_edges = []
    reverse_weights = []
    for edge, weight in zip(edges, weights):
        reverse_edges.append((edge[1], edge[0]))
        reverse_weights.append(weight)
    edges.extend(reverse_edges)
    weights.extend(reverse_weights)
    
    # Normalize weights
    if weights:
        max_weight = max(weights)
        normalized_weights = [w / max_weight for w in weights]
    else:
        normalized_weights = []
    
    return edges, weights, normalized_weights

def build_local_edges(d):
    edges = []
    # For a polynomial of degree d, there are d+1 coefficients (indices 0 to d)
    for i in range(d):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    return edges

def build_coefficient_graph(a, d, R_max=10, delta=6):
    node_features = []
    for i in range(len(a)):
        features = extract_node_features(a[i], i, d, a)
        node_features.append(features)
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # 1. Local edges (Backbone)
    local_edges = build_local_edges(d)
    # Assign a baseline weight to local edges (e.g. 1.0 or adaptive)
    # Paper implies these are structural backbone. 
    # We will treat them as having weight 1.0, which will be added to any Hankel overlap.
    local_weights = [1.0] * len(local_edges)
    
    # 2. Hankel edges (Multiplicity weighted)
    hankel_edges, hankel_weights, _ = build_hankel_edges(d, R_max, delta)
    
    # 3. Merge and sum weights correctly
    all_edges = local_edges + hankel_edges
    all_raw_weights = local_weights + hankel_weights
    
    unique_edges, all_weights, normalized_all_weights = deduplicate_and_weight_weighted(all_edges, all_raw_weights)
    
    edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(normalized_all_weights, dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)
    return data

def build_batch_graphs(dataset, R_max=10, delta=6, normalize_features=True, stats=None):
    graphs = []
    
    # Pre-calculate device for acceleration if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device} for feature statistics calculation")

    # First pass: collect all features if we need to compute stats
    if normalize_features and stats is None:
        # Collect features in chunks to avoid OOM
        feature_chunks = []
        chunk_size = 10000
        current_chunk = []
        
        for i, sample in enumerate(tqdm(dataset, desc="Collecting features for normalization")):
            a = sample['a']
            d = sample['d']
            graph = build_coefficient_graph(a, d, R_max, delta)
            current_chunk.append(graph.x)
            
            if len(current_chunk) >= chunk_size:
                feature_chunks.append(torch.cat(current_chunk, dim=0))
                current_chunk = []
        
        if current_chunk:
            feature_chunks.append(torch.cat(current_chunk, dim=0))
            
        # Compute stats using Welford's online algorithm or simple chunked aggregation
        # Here we use a simplified approach: compute sum and sum_sq on GPU per chunk
        total_sum = torch.zeros(3, device=device)
        total_sq_sum = torch.zeros(3, device=device)
        total_count = 0
        
        for chunk in tqdm(feature_chunks, desc="Computing stats on GPU"):
            chunk_gpu = chunk.to(device)
            total_sum += chunk_gpu.sum(dim=0)
            total_sq_sum += (chunk_gpu ** 2).sum(dim=0)
            total_count += chunk.size(0)
            del chunk_gpu
        
        mean = (total_sum / total_count).cpu()
        # var = E[X^2] - (E[X])^2
        var = (total_sq_sum / total_count) - mean.to(device) ** 2
        std = torch.sqrt(var).cpu() + 1e-12
        
        stats = (mean, std)
        
        # Clear memory
        del feature_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Second pass: build graphs with normalized features
    for sample in tqdm(dataset, desc="Building graphs"):
        a = sample['a']
        d = sample['d']
        graph = build_coefficient_graph(a, d, R_max, delta)
        
        # Normalize features if needed
        if normalize_features and stats is not None:
            mean, std = stats
            graph.x = (graph.x - mean) / std
        
        graph.y_rank = torch.tensor([sample['true_rank'] - 1], dtype=torch.long)
        noise_level = float(sample['noise_level'])
        
        # Improved stability label definition based on paper best practices
        # 1. Check root separation (min_spacing)
        roots = sample.get('roots', [])
        min_spacing = 1.0  # Default if no roots or single root
        if len(roots) > 1:
            roots_complex = [complex(r) if not isinstance(r, (int, float, complex)) else r for r in roots]
            min_dist = float('inf')
            for i in range(len(roots_complex)):
                for j in range(i + 1, len(roots_complex)):
                    dist = abs(roots_complex[i] - roots_complex[j])
                    if dist < min_dist:
                        min_dist = dist
            min_spacing = min_dist
            
        # 2. Check noise level (and potentially weight condition if available)
        # Thresholds: separation >= 1e-3, noise <= 1e-3
        is_stable = (min_spacing >= 1e-3) and (noise_level <= 1e-3)
        
        graph.y_stability = torch.tensor([1.0 if is_stable else 0.0], dtype=torch.float32)
        graphs.append(graph)
    
    return graphs, stats