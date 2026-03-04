import numpy as np
import torch
import time
import yaml
import os
import pandas as pd
from tqdm import tqdm
from data_generation import generate_sample_gpu, generate_sample
from solver import classical_sylvester, hybrid_solver, one_shot_reconstruction, build_hankel_matrix, solve_recurrence_coefficients, find_roots, solve_vandermonde, compute_residual_normalized_l2, compute_residual
from models import RankGNN
from graph_builder import build_coefficient_graph
from sklearn.metrics import f1_score
from evaluate import train_meta_classifier, evaluate_method_meta_solver

# Load Configuration
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Helper to get device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper to load model
def load_gnn_model(config, device):
    gnn_model = RankGNN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    )
    gnn_model = gnn_model.to(device)
    model_save_path = config['training']['model_save_path']
    if os.path.exists(model_save_path):
        gnn_model.load_state_dict(torch.load(model_save_path, map_location=device))
        gnn_model.eval()
        return gnn_model
    else:
        raise FileNotFoundError(f"Model file not found at {model_save_path}")

# Helper to generate dataset
def get_dataset(d, n_samples, noise_level, r_range, device):
    dataset = []
    # Generate in batches if possible, but for now loop is fine with GPU accel
    for _ in range(n_samples):
        r = np.random.randint(r_range[0], r_range[1] + 1)
        a, raw_a, true_rank, roots, weights = generate_sample_gpu(d, r, noise_level, device=device)
        dataset.append({
            'a': a,
            'true_rank': true_rank,
            'd': d,
            'noise_level': noise_level
        })
    return dataset

# -----------------------------------------------------------------------------
# Experiment 1: Rank Identification Accuracy (Table 5)
# -----------------------------------------------------------------------------
def run_rank_identification_experiment(config, gnn_model, meta_classifiers, scalers, stats, degrees=[50, 100, 200]):
    print("\n" + "="*50)
    print("Running Experiment 1: Rank Identification Accuracy (Table 5)")
    print("="*50)
    
    results = []
    n_samples = 2000 # Increased to 2000
    noise_level = 0.0 # Noiseless for Table 5
    R_max = config['model']['R_max']
    tau = 1e-6 # Strict tau for noiseless
    device = get_device()
    
    meta_clf_neighbor, meta_clf_top3, meta_clf_all = meta_classifiers
    scaler_neighbor, scaler_top3, scaler_all = scalers
    
    for d in degrees:
        print(f"Testing Degree d={d}...")
        dataset = get_dataset(d, n_samples, noise_level, config['data']['r_range'], device)
        
        # 1. Classical
        acc_c, f1_c, vsr_c = 0, 0, 0
        preds_c, trues_c = [], []
        vsr_count_c = 0
        
        for sample in tqdm(dataset, desc=f"Classical d={d}"):
            res = classical_sylvester(sample['a'], R_max, tau, metric='eps_2')
            if res:
                r_pred, _, _ = res
                vsr_count_c += 1
            else:
                r_pred = 0 # Failed
            preds_c.append(r_pred)
            trues_c.append(sample['true_rank'])
            
        acc_c = np.mean(np.array(preds_c) == np.array(trues_c))
        f1_c = f1_score(trues_c, preds_c, average='macro')
        vsr_c = vsr_count_c / n_samples
        
        results.append({
            'Degree': d,
            'Method': 'Classical',
            'Acc (%)': acc_c * 100,
            'F1 (%)': f1_c * 100,
            'VSR (%)': vsr_c * 100
        })

        # 2. Hybrid Solver (Original)
        acc_h, f1_h, vsr_h = 0, 0, 0
        preds_h, trues_h = [], []
        vsr_count_h = 0
        
        for sample in tqdm(dataset, desc=f"Hybrid d={d}"):
            res = hybrid_solver(sample['a'], sample['d'], gnn_model, R_max, base_tau=tau)
            if res:
                r_pred, _, _ = res
                vsr_count_h += 1
            else:
                r_pred = 0 # Failed
            preds_h.append(r_pred)
            trues_h.append(sample['true_rank'])
            
        acc_h = np.mean(np.array(preds_h) == np.array(trues_h))
        f1_h = f1_score(trues_h, preds_h, average='macro')
        vsr_h = vsr_count_h / n_samples
        
        results.append({
            'Degree': d,
            'Method': 'Hybrid (Original)',
            'Acc (%)': acc_h * 100,
            'F1 (%)': f1_h * 100,
            'VSR (%)': vsr_h * 100
        })

        # 3. Meta-Solver (Neighbor)
        meta_metrics_n = evaluate_method_meta_solver(
            gnn_model, dataset, meta_clf_neighbor, scaler_neighbor, R_max, stats, 
            strategy='neighbor', fallback_strategy='reject'
        )
        results.append({
            'Degree': d,
            'Method': 'Meta (Neighbor)',
            'Acc (%)': meta_metrics_n['accuracy'] * 100,
            'F1 (%)': meta_metrics_n['macro_f1'] * 100,
            'VSR (%)': meta_metrics_n['vsr'] * 100
        })

        # 4. Meta-Solver (Top-3)
        meta_metrics_t = evaluate_method_meta_solver(
            gnn_model, dataset, meta_clf_top3, scaler_top3, R_max, stats, 
            strategy='top3', fallback_strategy='reject'
        )
        results.append({
            'Degree': d,
            'Method': 'Meta (Top-3)',
            'Acc (%)': meta_metrics_t['accuracy'] * 100,
            'F1 (%)': meta_metrics_t['macro_f1'] * 100,
            'VSR (%)': meta_metrics_t['vsr'] * 100
        })

        # 5. Meta-Solver (All)
        meta_metrics_a = evaluate_method_meta_solver(
            gnn_model, dataset, meta_clf_all, scaler_all, R_max, stats, 
            strategy='all', fallback_strategy='reject'
        )
        results.append({
            'Degree': d,
            'Method': 'Meta (All)',
            'Acc (%)': meta_metrics_a['accuracy'] * 100,
            'F1 (%)': meta_metrics_a['macro_f1'] * 100,
            'VSR (%)': meta_metrics_a['vsr'] * 100
        })
        
    df = pd.DataFrame(results)
    print("\nTable 5 Results:")
    print(df)
    return df

# -----------------------------------------------------------------------------
# Experiment 2: Runtime Comparison (Table 6)
# -----------------------------------------------------------------------------
def run_runtime_comparison(config, gnn_model, meta_classifiers, scalers, stats, degrees=[50, 100, 200]):
    print("\n" + "="*50)
    print("Running Experiment 2: Runtime Comparison (Table 6)")
    print("="*50)
    
    results = []
    n_samples = 50 # Keep small for timing
    noise_level = 0.0
    R_max = config['model']['R_max']
    tau = 1e-6
    device = get_device()
    
    meta_clf_neighbor, meta_clf_top3, meta_clf_all = meta_classifiers
    scaler_neighbor, scaler_top3, scaler_all = scalers

    for d in degrees:
        print(f"Timing Degree d={d}...")
        dataset = get_dataset(d, n_samples, noise_level, config['data']['r_range'], device)
        
        # Classical Time
        times_c = []
        for sample in dataset:
            start = time.time()
            classical_sylvester(sample['a'], R_max, tau, metric='eps_2')
            times_c.append((time.time() - start) * 1000) # ms
        avg_time_c = np.mean(times_c)
        results.append({'Degree': d, 'Method': 'Classical', 'Time (ms)': avg_time_c})

        # Hybrid Time
        times_h = []
        for sample in dataset:
            start = time.time()
            hybrid_solver(sample['a'], sample['d'], gnn_model, R_max, base_tau=tau)
            times_h.append((time.time() - start) * 1000) # ms
        avg_time_h = np.mean(times_h)
        results.append({'Degree': d, 'Method': 'Hybrid', 'Time (ms)': avg_time_h})

        # Meta-Neighbor Time
        meta_metrics_n = evaluate_method_meta_solver(gnn_model, dataset, meta_clf_neighbor, scaler_neighbor, R_max, stats, strategy='neighbor', fallback_strategy='reject')
        avg_time_n = meta_metrics_n['avg_runtime'] * 1000
        results.append({'Degree': d, 'Method': 'Meta (Neighbor)', 'Time (ms)': avg_time_n})

        # Meta-Top3 Time
        meta_metrics_t = evaluate_method_meta_solver(gnn_model, dataset, meta_clf_top3, scaler_top3, R_max, stats, strategy='top3', fallback_strategy='reject')
        avg_time_t = meta_metrics_t['avg_runtime'] * 1000
        results.append({'Degree': d, 'Method': 'Meta (Top-3)', 'Time (ms)': avg_time_t})

        # Meta-All Time
        meta_metrics_a = evaluate_method_meta_solver(gnn_model, dataset, meta_clf_all, scaler_all, R_max, stats, strategy='all', fallback_strategy='reject')
        avg_time_a = meta_metrics_a['avg_runtime'] * 1000
        results.append({'Degree': d, 'Method': 'Meta (All)', 'Time (ms)': avg_time_a})
        
    df = pd.DataFrame(results)
    print("\nTable 6 Results:")
    print(df)
    return df

# -----------------------------------------------------------------------------
# Experiment 3: Noise Robustness (Figure 4)
# -----------------------------------------------------------------------------
def run_noise_robustness_experiment(config, gnn_model, meta_classifiers, scalers, stats):
    print("\n" + "="*50)
    print("Running Experiment 3: Noise Robustness (Figure 4)")
    print("="*50)
    
    d = 200
    noise_levels = [0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    n_samples = 2000 # Increased to 2000
    R_max = config['model']['R_max']
    # tau is adaptive inside loop
    
    device = get_device()
    results = []
    
    meta_clf_neighbor, meta_clf_top3, meta_clf_all = meta_classifiers
    scaler_neighbor, scaler_top3, scaler_all = scalers

    for sigma in noise_levels:
        print(f"Testing Noise Level sigma={sigma}...")
        dataset = get_dataset(d, n_samples, sigma, config['data']['r_range'], device)
        
        current_tau = 1e-6 if sigma <= 1e-4 else 1e-2
        
        # 1. Classical
        acc_c, vsr_count_c = 0, 0
        eps2_c = []
        preds_c, trues_c = [], []
        for sample in dataset:
            res = classical_sylvester(sample['a'], R_max, tau=current_tau, metric='eps_2')
            if res:
                r_pred, roots, weights = res
                vsr_count_c += 1
                eps2 = compute_residual_normalized_l2(sample['a'], roots, weights)
                eps2_c.append(eps2)
            else:
                r_pred = 0
            preds_c.append(r_pred)
            trues_c.append(sample['true_rank'])
        acc_c = np.mean(np.array(preds_c) == np.array(trues_c))
        vsr_c = vsr_count_c / n_samples
        results.append({'Noise': sigma, 'Method': 'Classical', 'Acc (%)': acc_c * 100, 'VSR (%)': vsr_c * 100})

        # 2. Hybrid (Original)
        acc_h, vsr_count_h = 0, 0
        preds_h, trues_h = [], []
        for sample in dataset:
            res = hybrid_solver(sample['a'], sample['d'], gnn_model, R_max, base_tau=current_tau)
            if res:
                r_pred, _, _ = res
                vsr_count_h += 1
            else:
                r_pred = 0
            preds_h.append(r_pred)
            trues_h.append(sample['true_rank'])
        acc_h = np.mean(np.array(preds_h) == np.array(trues_h))
        vsr_h = vsr_count_h / n_samples
        results.append({'Noise': sigma, 'Method': 'Hybrid', 'Acc (%)': acc_h * 100, 'VSR (%)': vsr_h * 100})

        # 3. Meta-Neighbor
        meta_metrics_n = evaluate_method_meta_solver(gnn_model, dataset, meta_clf_neighbor, scaler_neighbor, R_max, stats, strategy='neighbor', fallback_strategy='reject')
        results.append({'Noise': sigma, 'Method': 'Meta (Neighbor)', 'Acc (%)': meta_metrics_n['accuracy'] * 100, 'VSR (%)': meta_metrics_n['vsr'] * 100})

        # 4. Meta-Top3
        meta_metrics_t = evaluate_method_meta_solver(gnn_model, dataset, meta_clf_top3, scaler_top3, R_max, stats, strategy='top3', fallback_strategy='reject')
        results.append({'Noise': sigma, 'Method': 'Meta (Top-3)', 'Acc (%)': meta_metrics_t['accuracy'] * 100, 'VSR (%)': meta_metrics_t['vsr'] * 100})

        # 5. Meta-All
        meta_metrics_a = evaluate_method_meta_solver(gnn_model, dataset, meta_clf_all, scaler_all, R_max, stats, strategy='all', fallback_strategy='reject')
        results.append({'Noise': sigma, 'Method': 'Meta (All)', 'Acc (%)': meta_metrics_a['accuracy'] * 100, 'VSR (%)': meta_metrics_a['vsr'] * 100})
        
    df = pd.DataFrame(results)
    print("\nFigure 4 Data:")
    print(df)
    return df

def main():
    config = load_config('config.yaml')
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load Resources
    gnn_model = load_gnn_model(config, device)
    
    stats = None
    if os.path.exists('train_stats.pt'):
        stats = torch.load('train_stats.pt')
        print("Loaded training stats.")
        
    # 2. Train Meta Classifiers (All strategies)
    print("Generating validation set for Meta-Classifier training...")
    val_dataset = get_dataset(d=100, n_samples=500, noise_level=1e-4, r_range=config['data']['r_range'], device=device)
    
    print("Training Meta-Classifier (Neighbor Strategy)...")
    clf_n, scl_n = train_meta_classifier(gnn_model, val_dataset, config['model']['R_max'], stats, strategy='neighbor')
    
    print("Training Meta-Classifier (Top-3 Strategy)...")
    clf_t, scl_t = train_meta_classifier(gnn_model, val_dataset, config['model']['R_max'], stats, strategy='top3')
    
    print("Training Meta-Classifier (All Strategy)...")
    clf_a, scl_a = train_meta_classifier(gnn_model, val_dataset, config['model']['R_max'], stats, strategy='all')
    
    if None in [clf_n, clf_t, clf_a]:
        print("Failed to train one or more meta-classifiers. Exiting.")
        return

    meta_classifiers = [clf_n, clf_t, clf_a]
    scalers = [scl_n, scl_t, scl_a]

    # 3. Run Experiments
    run_rank_identification_experiment(config, gnn_model, meta_classifiers, scalers, stats)
    run_runtime_comparison(config, gnn_model, meta_classifiers, scalers, stats)
    run_noise_robustness_experiment(config, gnn_model, meta_classifiers, scalers, stats)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
