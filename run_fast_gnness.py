import yaml
import os
import sys
import torch

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from evaluate import evaluate_all_methods, evaluate_by_degree

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("Loading configuration...")
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    config = load_config(config_path)

    # Modify config for fast run
    print("Modifying configuration for fast run...")
    config['data']['train_samples'] = 200
    config['data']['val_samples'] = 50
    config['data']['test_samples'] = 50
    config['training']['n_epochs'] = 2
    config['training']['batch_size'] = 32
    config['training']['model_save_path'] = './models/rank_gnn_fast.pth'
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(config['training']['model_save_path']), exist_ok=True)

    print("Starting training...")
    try:
        train_model(config)
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Starting evaluation...")
    try:
        # Evaluate all methods
        print("\n--- Evaluating All Methods ---")
        evaluate_all_methods(config)
        
        # Evaluate by degree
        print("\n--- Evaluating By Degree ---")
        evaluate_by_degree(config)
        
        print("Evaluation completed successfully.")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
