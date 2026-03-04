import numpy as np
import torch
import yaml
import os
import pandas as pd
from tqdm import tqdm
from data_generation import generate_sample_gpu
from solver import classical_sylvester, hybrid_solver
from models import RankGNN
from evaluate import train_meta_classifier, evaluate_method_meta_solver
from sklearn.metrics import f1_score

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

def run_experiment(config, gnn_model, stats, degrees=[50, 100, 200]):
    print("\n" + "="*50)
    print("Running Experiment: MetaSolver with vs without GNN features")
    print("="*50)
    
    device = get_device()
    
    # 1. Train Meta Classifiers
    print("Generating validation set for Meta-Classifier training...")
    val_dataset = get_dataset(d=100, n_samples=500, noise_level=1e-4, r_range=config['data']['r_range'], device=device)
    
    strategies = ['neighbor', 'top3', 'all']
    classifiers = {}
    
    for strategy in strategies:
        print(f"\nTraining Meta-Classifier (Strategy: {strategy})...")
        
        # Train WITH GNN features
        print("  - With GNN features...")
        clf_with, scl_with = train_meta_classifier(gnn_model, val_dataset, config['model']['R_max'], stats, strategy=strategy, use_gnn_features=True)
        
        # Train WITHOUT GNN features
        print("  - Without GNN features...")
        clf_without, scl_without = train_meta_classifier(gnn_model, val_dataset, config['model']['R_max'], stats, strategy=strategy, use_gnn_features=False)
        
        classifiers[strategy] = {
            'with': (clf_with, scl_with),
            'without': (clf_without, scl_without)
        }

    # 2. Evaluate
    results = []
    n_samples = 1000
    noise_level = 0.0
    R_max = config['model']['R_max']
    
    for d in degrees:
        print(f"\nTesting Degree d={d}...")
        dataset = get_dataset(d, n_samples, noise_level, config['data']['r_range'], device)
        
        for strategy in strategies:
            # Evaluate WITH GNN features
            clf, scl = classifiers[strategy]['with']
            if clf:
                metrics = evaluate_method_meta_solver(
                    gnn_model, dataset, clf, scl, R_max, stats, 
                    strategy=strategy, fallback_strategy='reject', use_gnn_features=True
                )
                results.append({
                    'Degree': d,
                    'Method': f'Meta ({strategy}) - With GNN Feat',
                    'Acc (%)': metrics['accuracy'] * 100,
                    'VSR (%)': metrics['vsr'] * 100
                })
            
            # Evaluate WITHOUT GNN features
            clf, scl = classifiers[strategy]['without']
            if clf:
                metrics = evaluate_method_meta_solver(
                    gnn_model, dataset, clf, scl, R_max, stats, 
                    strategy=strategy, fallback_strategy='reject', use_gnn_features=False
                )
                results.append({
                    'Degree': d,
                    'Method': f'Meta ({strategy}) - No GNN Feat',
                    'Acc (%)': metrics['accuracy'] * 100,
                    'VSR (%)': metrics['vsr'] * 100
                })

    df = pd.DataFrame(results)
    print("\nResults:")
    print(df)
    return df

def main():
    config = load_config('config.yaml')
    device = get_device()
    print(f"Using device: {device}")
    
    gnn_model = load_gnn_model(config, device)
    
    stats = None
    if os.path.exists('train_stats.pt'):
        stats = torch.load('train_stats.pt')
        print("Loaded training stats.")
        
    run_experiment(config, gnn_model, stats)

if __name__ == "__main__":
    main()
