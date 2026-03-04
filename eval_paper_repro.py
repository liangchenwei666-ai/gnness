import yaml
import os
import sys
import torch
import time

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_all_methods, evaluate_by_degree

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("Loading configuration for evaluation...")
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    config = load_config(config_path)

    # Use the paper model path
    config['training']['model_save_path'] = './models/rank_gnn_paper.pth'
    
    # Paper parameters (for consistent R_max and tau)
    config['model']['R_max'] = 10
    config['solver']['tau'] = 1e-3 # Or use 1e-6 as in paper, but hybrid uses dynamic. Let's stick to config or update it.
    
    # Ensure model exists
    if not os.path.exists(config['training']['model_save_path']):
        print(f"Error: Model file {config['training']['model_save_path']} not found. Run training first.")
        return

    # Use the same test sample size as training script
    config['data']['test_samples'] = 3000

    print("\n--- Starting Evaluation ---")
    try:
        # Evaluate all methods
        print("\nEvaluating All Methods...")
        evaluate_all_methods(config)
        
        # Evaluate by degree
        print("\nEvaluating By Degree...")
        evaluate_by_degree(config)
        
        print("Evaluation completed successfully.")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
