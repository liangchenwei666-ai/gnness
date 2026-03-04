import numpy as np
import torch
import time
from sklearn.metrics import f1_score
from data_generation import generate_dataset, split_dataset_with_balance
from solver import classical_sylvester, svd_baseline, hybrid_solver
from models import RankGNN
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_residuals(a, roots, weights):
    n = len(a)
    predicted = np.sum(weights[:, None] * (roots[:, None] ** np.arange(n)), axis=0)
    eps_max = np.max(np.abs(a - predicted))
    eps = 1e-12
    norm_a = np.linalg.norm(a) + eps
    eps_2 = np.linalg.norm(a - predicted) / norm_a
    return eps_max, eps_2

from sklearn.metrics import top_k_accuracy_score

from sklearn.metrics import f1_score
from solver import classical_sylvester, svd_baseline, hybrid_solver, compute_residual, build_hankel_matrix, solve_recurrence_coefficients, find_roots, solve_vandermonde, gnn_predict_rank_with_stability, one_shot_reconstruction
from graph_builder import build_coefficient_graph
from sklearn.metrics import top_k_accuracy_score

def evaluate_method_gnn_direct(gnn_model, test_dataset, R_max=10, stats=None):
    metrics = {
        'rank_acc': 0.0,
        'rank_acc_top3': 0.0,
        'rank_within_1': 0.0,
        'pred_dist': {},
        'true_dist': {},
        'rank_preds': [],
        'true_ranks': [],
        'stability_scores': [],
        'confidence_scores': []
    }
    
    gnn_model.eval()
    device = next(gnn_model.parameters()).device
    
    for sample in test_dataset:
        a = sample['a']
        d = sample['d']
        true_rank = sample['true_rank']
        
        # Build graph on-the-fly (CPU)
        graph = build_coefficient_graph(a, d, R_max)
        
        # Apply normalization if stats provided
        if stats is not None:
             mean, std = stats
             graph.x = (graph.x - mean) / std
        
        # Predict on GPU
        with torch.no_grad():
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            edge_weight = graph.edge_attr.to(device) if hasattr(graph, 'edge_attr') else None
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
            
            rank_logits, stability_score = gnn_model(x, edge_index, batch, edge_weight)
            probs = torch.softmax(rank_logits, dim=1).cpu().numpy()[0]
            
            # Predictions
            r_pred_idx = np.argmax(probs)
            r_pred = r_pred_idx + 1
            confidence = probs[r_pred_idx]
            
            # Collect metrics
            metrics['rank_preds'].append(r_pred)
            metrics['true_ranks'].append(true_rank)
            metrics['stability_scores'].append(stability_score.item())
            metrics['confidence_scores'].append(confidence)
            
            # Accuracy
            if r_pred == true_rank:
                metrics['rank_acc'] += 1
                
            # Within +/- 1
            if abs(r_pred - true_rank) <= 1:
                metrics['rank_within_1'] += 1
                
            # Distributions
            metrics['pred_dist'][r_pred] = metrics['pred_dist'].get(r_pred, 0) + 1
            metrics['true_dist'][true_rank] = metrics['true_dist'].get(true_rank, 0) + 1
            
            # For Top-k, we need batch format but here we do one by one. 
            # Simple check: is true_rank-1 in top 3 indices?
            top3_indices = np.argsort(probs)[-3:]
            if (true_rank - 1) in top3_indices:
                metrics['rank_acc_top3'] += 1

    total = len(test_dataset)
    metrics['rank_acc'] /= total
    metrics['rank_within_1'] /= total
    metrics['rank_acc_top3'] /= total
    
    return metrics

def train_meta_classifier(gnn_model, val_dataset, R_max=10, stats=None, strategy='neighbor', use_gnn_features=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    gnn_model.eval()
    device = next(gnn_model.parameters()).device
    
    X = []
    y = []
    
    print(f"Collecting data for meta-classifier training (Strategy: {strategy}, use_gnn_features={use_gnn_features})...")
    for sample in val_dataset:
        a = sample['a']
        d = sample['d']
        true_rank = sample['true_rank']
        
        graph = build_coefficient_graph(a, d, R_max)
        if stats is not None:
             mean, std = stats
             graph.x = (graph.x - mean) / std
             
        with torch.no_grad():
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            edge_weight = graph.edge_attr.to(device) if hasattr(graph, 'edge_attr') else None
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
            rank_logits, stability_score = gnn_model(x, edge_index, batch, edge_weight)
            probs = torch.softmax(rank_logits, dim=1).cpu().numpy()[0]
            
        r_pred_idx = np.argmax(probs)
        r_pred = r_pred_idx + 1
        confidence = probs[r_pred_idx]
        stability = stability_score.item()
        
        candidates = []
        if strategy == 'neighbor':
            # r_pred, r_pred-1, r_pred+1
            candidates = [r_pred, r_pred-1, r_pred+1]
        elif strategy == 'top3':
            # Top-3 predictions
            top3_idx = np.argsort(probs)[-3:][::-1]
            candidates = [idx + 1 for idx in top3_idx]
        elif strategy == 'all':
            # All possible ranks 1..R_max
            candidates = list(range(1, R_max + 1))
            
        candidates = sorted(list(set([c for c in candidates if 1 <= c <= R_max])))
        
        for r_c in candidates:
            # We need to pass the specific confidence for this candidate r_c
            # The original confidence was just for r_pred (max). 
            # Now we fetch prob for r_c
            prob_c = probs[r_c - 1]
            
            res = one_shot_reconstruction(a, r_c, tau=1.0) 
            if res is not None:
                _, roots, weights = res
                eps_max, eps_2 = compute_residual(a, roots, weights), 0 
                
                H = build_hankel_matrix(a, r_c + 1)
                c = solve_recurrence_coefficients(H)
                if len(c) == r_c:
                    roots = find_roots(c)
                    if len(roots) == r_c:
                        weights = solve_vandermonde(roots, a)
                        eps_max_val = compute_residual(a, roots, weights)
                        pred_a = np.sum(weights[:, None] * (roots[:, None] ** np.arange(len(a))), axis=0)
                        norm_a = np.linalg.norm(a) + 1e-12
                        eps_2_val = np.linalg.norm(a - pred_a) / norm_a
                        
                        # Feature: Use prob_c instead of general confidence?
                        # Or keep confidence of top-1 as a global quality metric?
                        # Let's use prob_c (specific confidence) as it's more informative for "all" strategy
                        if use_gnn_features:
                            features = [np.log10(eps_max_val + 1e-16), np.log10(eps_2_val + 1e-16), stability, prob_c, abs(r_c - r_pred)]
                        else:
                            features = [np.log10(eps_max_val + 1e-16), np.log10(eps_2_val + 1e-16)]
                        X.append(features)
                        y.append(1 if r_c == true_rank else 0)

    if not X:
        print("No valid reconstruction samples for meta-training.")
        return None, None

    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_scaled, y)
    
    print(f"Meta-classifier trained. Acc: {clf.score(X_scaled, y):.4f}")
    return clf, scaler

def evaluate_method_meta_solver(gnn_model, test_dataset, meta_clf, scaler, R_max=10, stats=None, strategy='neighbor', fallback_strategy='reject', use_gnn_features=True):
    # fallback_strategy: 'reject' (output 0), 'best_prob' (output candidate with highest prob even if < 0.5), 'gnn_raw' (output r_pred)
    
    metrics = {
        'accuracy': 0.0,
        'vsr': 0.0,
        'runtime': [],
        'eps_max': [],
        'eps_2': [],
        'rank_preds': [],
        'true_ranks': []
    }
    
    gnn_model.eval()
    device = next(gnn_model.parameters()).device
    
    for sample in test_dataset:
        a = sample['a']
        d = sample['d']
        true_rank = sample['true_rank']
        start_time = time.time()
        
        # 1. GNN Prediction
        graph = build_coefficient_graph(a, d, R_max)
        if stats is not None:
             mean, std = stats
             graph.x = (graph.x - mean) / std
             
        with torch.no_grad():
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            edge_weight = graph.edge_attr.to(device) if hasattr(graph, 'edge_attr') else None
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
            rank_logits, stability_score = gnn_model(x, edge_index, batch, edge_weight)
            probs = torch.softmax(rank_logits, dim=1).cpu().numpy()[0]
        
        r_pred_idx = np.argmax(probs)
        r_pred = r_pred_idx + 1
        confidence = probs[r_pred_idx]
        stability = stability_score.item()
        
        # 2. Candidate Evaluation
        candidates = []
        if strategy == 'neighbor':
            candidates = [r_pred, r_pred-1, r_pred+1]
        elif strategy == 'top3':
            top3_idx = np.argsort(probs)[-3:][::-1]
            candidates = [idx + 1 for idx in top3_idx]
        elif strategy == 'all':
            candidates = list(range(1, R_max + 1))
            
        candidates = sorted(list(set([c for c in candidates if 1 <= c <= R_max])))
        
        best_candidate = None
        best_prob = -1.0
        best_eps_max = 0.0
        best_eps_2 = 0.0
        
        for r_c in candidates:
            prob_c = probs[r_c - 1]
            
             # Reconstruction
            H = build_hankel_matrix(a, r_c + 1)
            c = solve_recurrence_coefficients(H)
            if len(c) == r_c:
                roots = find_roots(c)
                if len(roots) == r_c:
                    weights = solve_vandermonde(roots, a)
                    eps_max_val = compute_residual(a, roots, weights)
                    
                    pred_a = np.sum(weights[:, None] * (roots[:, None] ** np.arange(len(a))), axis=0)
                    norm_a = np.linalg.norm(a) + 1e-12
                    eps_2_val = np.linalg.norm(a - pred_a) / norm_a
                    
                    # Meta-feature construction
                    if use_gnn_features:
                        features = np.array([[np.log10(eps_max_val + 1e-16), np.log10(eps_2_val + 1e-16), stability, prob_c, abs(r_c - r_pred)]])
                    else:
                        features = np.array([[np.log10(eps_max_val + 1e-16), np.log10(eps_2_val + 1e-16)]])
                    features_scaled = scaler.transform(features)
                    
                    # Probability of being correct
                    prob_correct = meta_clf.predict_proba(features_scaled)[0, 1]
                    
                    if prob_correct > best_prob:
                        best_prob = prob_correct
                        best_candidate = r_c
                        best_eps_max = eps_max_val
                        best_eps_2 = eps_2_val
        
        end_time = time.time()
        metrics['runtime'].append(end_time - start_time)
        
        # Decision
        final_pred = 0 
        verified = False
        
        if best_candidate is not None and best_prob > 0.5: # Threshold 0.5
            final_pred = best_candidate
            verified = True
        else:
             # Fallback logic
             if fallback_strategy == 'reject':
                 final_pred = 0
             elif fallback_strategy == 'best_prob':
                 final_pred = best_candidate if best_candidate is not None else r_pred # fallback to r_pred if no reconstruction succeeded
             elif fallback_strategy == 'gnn_raw':
                 final_pred = r_pred
        
        if verified:
            metrics['vsr'] += 1
            metrics['eps_max'].append(best_eps_max)
            metrics['eps_2'].append(best_eps_2)
        
        if final_pred != 0:
            metrics['coverage'] = metrics.get('coverage', 0) + 1
            
        metrics['rank_preds'].append(final_pred)
        metrics['true_ranks'].append(true_rank)
        
        if final_pred == true_rank:
            metrics['accuracy'] += 1
            
    total = len(test_dataset)
    metrics['accuracy'] /= total
    metrics['vsr'] /= total
    metrics['coverage'] = metrics.get('coverage', 0) / total
    metrics['avg_runtime'] = np.mean(metrics['runtime'])
    metrics['std_runtime'] = np.std(metrics['runtime'])
    metrics['avg_eps_max'] = np.mean(metrics['eps_max']) if metrics['eps_max'] else 0
    metrics['std_eps_max'] = np.std(metrics['eps_max']) if metrics['eps_max'] else 0
    metrics['avg_eps_2'] = np.mean(metrics['eps_2']) if metrics['eps_2'] else 0
    metrics['std_eps_2'] = np.std(metrics['eps_2']) if metrics['eps_2'] else 0
    
    if metrics['rank_preds'] and metrics['true_ranks']:
        metrics['macro_f1'] = f1_score(metrics['true_ranks'], metrics['rank_preds'], average='macro')
    else:
        metrics['macro_f1'] = 0
    
    return metrics


def evaluate_by_sample_type(metrics_fn, test_dataset, sample_type, **kwargs):
    filtered_dataset = [sample for sample in test_dataset if sample['sample_type'] == sample_type]
    if not filtered_dataset:
        return None
    return metrics_fn(filtered_dataset, **kwargs)

def evaluate_method(method, test_dataset, R_max=10, tau=1e-6, gnn_model=None):
    metrics = {
        'accuracy': 0.0,
        'vsr': 0.0,
        'runtime': [],
        'eps_max': [],
        'eps_2': [],
        'rank_preds': [],
        'true_ranks': []
    }
    for sample in test_dataset:
        a = sample['a']
        true_rank = sample['true_rank']
        d = sample['d']
        start_time = time.time()
        if method == 'classical':
            result = classical_sylvester(a, R_max, tau)
        elif method == 'svd':
            result = svd_baseline(a, R_max, tau)
        elif method == 'hybrid':
            result = hybrid_solver(a, d, gnn_model, R_max, base_tau=tau)
        else:
            raise ValueError("Unsupported method")
        end_time = time.time()
        metrics['runtime'].append(end_time - start_time)
        metrics['true_ranks'].append(true_rank)
        if result is not None:
            r_pred, roots, weights = result
            metrics['rank_preds'].append(r_pred)
            metrics['vsr'] += 1
            if r_pred == true_rank:
                metrics['accuracy'] += 1
            eps_max, eps_2 = compute_residuals(a, roots, weights)
            metrics['eps_max'].append(eps_max)
            metrics['eps_2'].append(eps_2)
        else:
            metrics['rank_preds'].append(0)
    total_samples = len(test_dataset)
    metrics['accuracy'] /= total_samples
    metrics['vsr'] /= total_samples
    metrics['avg_runtime'] = np.mean(metrics['runtime'])
    metrics['std_runtime'] = np.std(metrics['runtime'])
    metrics['avg_eps_max'] = np.mean(metrics['eps_max']) if metrics['eps_max'] else 0
    metrics['std_eps_max'] = np.std(metrics['eps_max']) if metrics['eps_max'] else 0
    metrics['avg_eps_2'] = np.mean(metrics['eps_2']) if metrics['eps_2'] else 0
    metrics['std_eps_2'] = np.std(metrics['eps_2']) if metrics['eps_2'] else 0
    if metrics['rank_preds'] and metrics['true_ranks']:
        metrics['macro_f1'] = f1_score(metrics['true_ranks'], metrics['rank_preds'], average='macro')
    else:
        metrics['macro_f1'] = 0
    return metrics

def evaluate_true_rank_residuals(test_dataset):
    print("\nCalculating residuals for TRUE ranks (Oracle Solver)...")
    eps_max_list = []
    eps_2_list = []
    
    for sample in test_dataset:
        a = sample['a']
        true_rank = sample['true_rank']
        
        # Force reconstruction using the TRUE rank
        # We use a large tau to ensure we always get a result if mathematically possible
        # (i.e. if H is not singular)
        try:
            res = one_shot_reconstruction(a, true_rank, tau=1e9)
            if res is not None:
                _, roots, weights = res
                eps_max = compute_residual(a, roots, weights)
                
                # Compute eps_2 manually
                pred_a = np.sum(weights[:, None] * (roots[:, None] ** np.arange(len(a))), axis=0)
                norm_a = np.linalg.norm(a) + 1e-12
                eps_2 = np.linalg.norm(a - pred_a) / norm_a
                
                eps_max_list.append(eps_max)
                eps_2_list.append(eps_2)
            else:
                # If reconstruction fails (e.g. singular matrix), we might skip or record nan
                pass
        except Exception:
            pass
            
    avg_eps_max = np.mean(eps_max_list) if eps_max_list else 0.0
    avg_eps_2 = np.mean(eps_2_list) if eps_2_list else 0.0
    
    print(f"  True Rank Avg eps_max: {avg_eps_max:.4f}")
    print(f"  True Rank Avg eps_2: {avg_eps_2:.4f}")
    return avg_eps_max, avg_eps_2

def evaluate_all_methods(config, test_dataset=None):
    if test_dataset is None:
        test_dataset = generate_dataset(
            n_samples=config['data']['test_samples'],
            d_range=config['data']['d_range'],
            r_range=config['data']['r_range'],
            noise_levels=config['data']['noise_levels'],
            include_hard_samples=config['data']['include_hard_samples']
        )
    
    # Calculate Oracle Residuals first
    evaluate_true_rank_residuals(test_dataset)
    
    # Load training stats if available for proper normalization
    import os
    stats = None
    if os.path.exists('train_stats.pt'):
        stats = torch.load('train_stats.pt')
        print("Loaded training stats for normalization.")
    
    R_max = config['model']['R_max']
    tau = config['solver']['tau']
    gnn_model = RankGNN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    gnn_model = gnn_model.to(device)
    
    model_save_path = config['training']['model_save_path']
    gnn_model.load_state_dict(torch.load(model_save_path, map_location=device))
    gnn_model.eval()
    def eval_method_wrapper(dataset):
        if not dataset:
            empty_metrics = {
                'accuracy': 0.0,
                'vsr': 0.0,
                'runtime': [],
                'eps_max': [],
                'eps_2': [],
                'rank_preds': [],
                'true_ranks': [],
                'avg_runtime': 0.0,
                'std_runtime': 0.0,
                'avg_eps_max': 0.0,
                'std_eps_max': 0.0,
                'avg_eps_2': 0.0,
                'std_eps_2': 0.0,
                'macro_f1': 0.0
            }
            return empty_metrics, empty_metrics, empty_metrics
        return (
            evaluate_method('classical', dataset, R_max, tau),
            evaluate_method('svd', dataset, R_max, tau),
            evaluate_method('hybrid', dataset, R_max, tau, gnn_model)
        )
    print("\nEvaluating GNN Direct Prediction...")
    gnn_metrics = evaluate_method_gnn_direct(gnn_model, test_dataset, R_max, stats)
    print(f"  Rank Accuracy: {gnn_metrics['rank_acc']:.4f}")
    print(f"  Rank Accuracy (Top-3): {gnn_metrics['rank_acc_top3']:.4f}")
    print(f"  Rank Accuracy (within +/- 1): {gnn_metrics['rank_within_1']:.4f}")
    print("  Prediction Distribution:", dict(sorted(gnn_metrics['pred_dist'].items())))
    print("  True Rank Distribution:", dict(sorted(gnn_metrics['true_dist'].items())))
    
    # Train Meta-Classifier using VALIDATION set (to avoid overfitting on test)
    # We need to generate/load validation set. Since we don't have it loaded here, let's generate a small one or split test
    # Ideally, we should use the validation set from training.
    # For now, let's split test_dataset: first 20% for meta-train, rest for meta-test
    split_idx = int(len(test_dataset) * 0.2)
    val_subset = test_dataset[:split_idx]
    test_subset = test_dataset[split_idx:]
    
    print("\nTraining Meta-Classifier for Solver Decision (Strategy: neighbor)...")
    meta_clf_neighbor, scaler_neighbor = train_meta_classifier(gnn_model, val_subset, R_max, stats, strategy='neighbor')
    
    if meta_clf_neighbor is not None:
        print("\nEvaluating Meta-Learned Solver (neighbor strategy, fallback=reject)...")
        meta_metrics = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_neighbor, scaler_neighbor, R_max, stats, strategy='neighbor', fallback_strategy='reject')
        print(f"  Accuracy: {meta_metrics['accuracy']:.4f}")
        print(f"  Macro-F1: {meta_metrics['macro_f1']:.4f}")
        print(f"  VSR: {meta_metrics['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics['coverage']:.4f}")
        print(f"  Avg Runtime: {meta_metrics['avg_runtime']:.4f}s")
        
        print("\nEvaluating Meta-Learned Solver (neighbor strategy, fallback=best_prob)...")
        meta_metrics_best = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_neighbor, scaler_neighbor, R_max, stats, strategy='neighbor', fallback_strategy='best_prob')
        print(f"  Accuracy: {meta_metrics_best['accuracy']:.4f}")
        print(f"  VSR: {meta_metrics_best['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics_best['coverage']:.4f}")
        
        print("\nEvaluating Meta-Learned Solver (neighbor strategy, fallback=gnn_raw)...")
        meta_metrics_raw = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_neighbor, scaler_neighbor, R_max, stats, strategy='neighbor', fallback_strategy='gnn_raw')
        print(f"  Accuracy: {meta_metrics_raw['accuracy']:.4f}")
        print(f"  VSR: {meta_metrics_raw['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics_raw['coverage']:.4f}")
        
    print("\nTraining Meta-Classifier for Solver Decision (Strategy: top3)...")
    meta_clf_top3, scaler_top3 = train_meta_classifier(gnn_model, val_subset, R_max, stats, strategy='top3')
    
    if meta_clf_top3 is not None:
        print("\nEvaluating Meta-Learned Solver (top3 strategy, fallback=reject)...")
        meta_metrics = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_top3, scaler_top3, R_max, stats, strategy='top3', fallback_strategy='reject')
        print(f"  Accuracy: {meta_metrics['accuracy']:.4f}")
        print(f"  Macro-F1: {meta_metrics['macro_f1']:.4f}")
        print(f"  VSR: {meta_metrics['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics['coverage']:.4f}")
        print(f"  Avg Runtime: {meta_metrics['avg_runtime']:.4f}s")
        
        print("\nEvaluating Meta-Learned Solver (top3 strategy, fallback=best_prob)...")
        meta_metrics_best = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_top3, scaler_top3, R_max, stats, strategy='top3', fallback_strategy='best_prob')
        print(f"  Accuracy: {meta_metrics_best['accuracy']:.4f}")
        print(f"  VSR: {meta_metrics_best['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics_best['coverage']:.4f}")
        
        print("\nEvaluating Meta-Learned Solver (top3 strategy, fallback=gnn_raw)...")
        meta_metrics_raw = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_top3, scaler_top3, R_max, stats, strategy='top3', fallback_strategy='gnn_raw')
        print(f"  Accuracy: {meta_metrics_raw['accuracy']:.4f}")
        print(f"  VSR: {meta_metrics_raw['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics_raw['coverage']:.4f}")
        
    print("\nTraining Meta-Classifier for Solver Decision (Strategy: all)...")
    meta_clf_all, scaler_all = train_meta_classifier(gnn_model, val_subset, R_max, stats, strategy='all')
    
    if meta_clf_all is not None:
        print("\nEvaluating Meta-Learned Solver (all strategy, fallback=reject)...")
        meta_metrics = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_all, scaler_all, R_max, stats, strategy='all', fallback_strategy='reject')
        print(f"  Accuracy: {meta_metrics['accuracy']:.4f}")
        print(f"  Macro-F1: {meta_metrics['macro_f1']:.4f}")
        print(f"  VSR: {meta_metrics['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics['coverage']:.4f}")
        print(f"  Avg Runtime: {meta_metrics['avg_runtime']:.4f}s")
        
        print("\nEvaluating Meta-Learned Solver (all strategy, fallback=best_prob)...")
        meta_metrics_best = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_all, scaler_all, R_max, stats, strategy='all', fallback_strategy='best_prob')
        print(f"  Accuracy: {meta_metrics_best['accuracy']:.4f}")
        print(f"  VSR: {meta_metrics_best['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics_best['coverage']:.4f}")
        
        print("\nEvaluating Meta-Learned Solver (all strategy, fallback=gnn_raw)...")
        meta_metrics_raw = evaluate_method_meta_solver(gnn_model, test_subset, meta_clf_all, scaler_all, R_max, stats, strategy='all', fallback_strategy='gnn_raw')
        print(f"  Accuracy: {meta_metrics_raw['accuracy']:.4f}")
        print(f"  VSR: {meta_metrics_raw['vsr']:.4f}")
        print(f"  Coverage: {meta_metrics_raw['coverage']:.4f}")
    
    # Continue with original evaluation for comparison (on full set or subset?)
    # Let's run original on full set for consistency with previous runs
    print("\nEvaluating Original Methods (Classical/SVD/Hybrid)...")
    classical_metrics, svd_metrics, hybrid_metrics = eval_method_wrapper(test_dataset)
    print("Evaluating on standard samples...")
    standard_classical, standard_svd, standard_hybrid = eval_method_wrapper([s for s in test_dataset if s['sample_type'] == 'standard'])
    print("Evaluating on near-collision samples...")
    near_collision_classical, near_collision_svd, near_collision_hybrid = eval_method_wrapper([s for s in test_dataset if s['sample_type'] == 'near_collision'])
    print("Evaluating on ill-conditioned samples...")
    ill_conditioned_classical, ill_conditioned_svd, ill_conditioned_hybrid = eval_method_wrapper([s for s in test_dataset if s['sample_type'] == 'ill_conditioned'])
    print("\nEvaluation Results - All Samples:")
    print("===============================")
    print("Classical Sylvester:")
    print(f"  Accuracy: {classical_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {classical_metrics['macro_f1']:.4f}")
    print(f"  VSR: {classical_metrics['vsr']:.4f}")
    print(f"  Avg Runtime: {classical_metrics['avg_runtime']:.4f}s")
    print(f"  Avg eps_max: {classical_metrics['avg_eps_max']:.4f}")
    print(f"  Avg eps_2: {classical_metrics['avg_eps_2']:.4f}")
    print("\nSVD Baseline:")
    print(f"  Accuracy: {svd_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {svd_metrics['macro_f1']:.4f}")
    print(f"  VSR: {svd_metrics['vsr']:.4f}")
    print(f"  Avg Runtime: {svd_metrics['avg_runtime']:.4f}s")
    print(f"  Avg eps_max: {svd_metrics['avg_eps_max']:.4f}")
    print(f"  Avg eps_2: {svd_metrics['avg_eps_2']:.4f}")
    print("\nHybrid Solver:")
    print(f"  Accuracy: {hybrid_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {hybrid_metrics['macro_f1']:.4f}")
    print(f"  VSR: {hybrid_metrics['vsr']:.4f}")
    print(f"  Avg Runtime: {hybrid_metrics['avg_runtime']:.4f}s")
    print(f"  Avg eps_max: {hybrid_metrics['avg_eps_max']:.4f}")
    print(f"  Avg eps_2: {hybrid_metrics['avg_eps_2']:.4f}")
    return [meta_clf_neighbor, meta_clf_top3, meta_clf_all], [scaler_neighbor, scaler_top3, scaler_all]

from data_generation import generate_sample_gpu, generate_sample
import random
from tqdm import tqdm

def evaluate_by_degree(config, meta_clf_list=None, scaler_list=None, stats=None):
    # This function evaluates performance for each degree d in the range
    d_range = config['data']['d_range']
    n_samples_per_degree = 1000 # fixed for quick eval
    
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Accelerating data generation with {device}")
    
    # Load model
    gnn_model = RankGNN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    )
    print(f"Evaluating by degree on device: {device}")
    gnn_model = gnn_model.to(device)

    model_save_path = config['training']['model_save_path']
    gnn_model.load_state_dict(torch.load(model_save_path, map_location=device))
    gnn_model.eval()
    
    R_max = config['model']['R_max']
    tau = config['solver']['tau']

    results = {}

    for d in d_range:
        print(f"Evaluating for degree d={d}...")
        # Generate fresh samples for this degree
        degree_dataset = []
        for _ in tqdm(range(n_samples_per_degree), desc=f"Generating samples for degree {d}"):
             r = random.randint(config['data']['r_range'][0], config['data']['r_range'][1])
             noise_level = random.choice(config['data']['noise_levels'])
             if use_gpu:
                 a, raw_a, true_rank, roots, weights = generate_sample_gpu(d, r, noise_level, device=device)
             else:
                 a, raw_a, true_rank, roots, weights = generate_sample(d, r, noise_level)
             
             degree_dataset.append({
                'a': a,
                'raw_a': raw_a,
                'true_rank': true_rank,
                'roots': roots,
                'weights': weights,
                'd': d,
                'noise_level': noise_level,
                'sample_type': 'standard'
            })

        # Evaluate Classical
        metrics_classical = evaluate_method('classical', degree_dataset, R_max, tau)
        print(f"  Classical - Acc: {metrics_classical['accuracy']:.4f}, VSR: {metrics_classical['vsr']:.4f}, Time: {metrics_classical['avg_runtime']:.4f}s")
        
        # Evaluate SVD
        metrics_svd = evaluate_method('svd', degree_dataset, R_max, tau)
        print(f"  SVD - Acc: {metrics_svd['accuracy']:.4f}, VSR: {metrics_svd['vsr']:.4f}, Time: {metrics_svd['avg_runtime']:.4f}s")
        
        # Evaluate Hybrid
        metrics_hybrid = evaluate_method('hybrid', degree_dataset, R_max, tau, gnn_model)
        print(f"  Hybrid - Acc: {metrics_hybrid['accuracy']:.4f}, VSR: {metrics_hybrid['vsr']:.4f}, Time: {metrics_hybrid['avg_runtime']:.4f}s")
        
        # Evaluate Meta-Solver (Neighbor)
        if meta_clf_list is not None and len(meta_clf_list) >= 3:
             metrics_meta_neighbor = evaluate_method_meta_solver(gnn_model, degree_dataset, meta_clf_list[0], scaler_list[0], R_max, stats, strategy='neighbor')
             print(f"  Meta (Neighbor) - Acc: {metrics_meta_neighbor['accuracy']:.4f}, VSR: {metrics_meta_neighbor['vsr']:.4f}, Time: {metrics_meta_neighbor['avg_runtime']:.4f}s")
             
             metrics_meta_top3 = evaluate_method_meta_solver(gnn_model, degree_dataset, meta_clf_list[1], scaler_list[1], R_max, stats, strategy='top3')
             print(f"  Meta (Top3) - Acc: {metrics_meta_top3['accuracy']:.4f}, VSR: {metrics_meta_top3['vsr']:.4f}, Time: {metrics_meta_top3['avg_runtime']:.4f}s")
             
             metrics_meta_all = evaluate_method_meta_solver(gnn_model, degree_dataset, meta_clf_list[2], scaler_list[2], R_max, stats, strategy='all')
             print(f"  Meta (All) - Acc: {metrics_meta_all['accuracy']:.4f}, VSR: {metrics_meta_all['vsr']:.4f}, Time: {metrics_meta_all['avg_runtime']:.4f}s")
             
    return results

import os
def main():
    config = load_config('config.yaml')
    
    # First, run the main evaluation which returns the trained meta-classifiers
    # But wait, evaluate_all_methods doesn't return them currently. We need to modify it.
    # Let's just modify evaluate_all_methods to return them.
    
    # Or simpler: Just extract the training logic out.
    # But for now, let's just make evaluate_all_methods return the classifiers.
    meta_classifiers, scalers = evaluate_all_methods(config)
    
    # Then run by degree evaluation using those classifiers
    if meta_classifiers:
        evaluate_by_degree(config, meta_classifiers, scalers, stats=torch.load('train_stats.pt') if os.path.exists('train_stats.pt') else None)
    else:
        evaluate_by_degree(config)
        
    print('Evaluation completed!')

if __name__ == '__main__':
    main()