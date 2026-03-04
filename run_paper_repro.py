import yaml
import os
import sys
import torch
import time

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

    # Modify config to match paper's training parameters more closely
    print("Setting training parameters according to paper (Table 3)...")
    
    # Paper parameters
    config['training']['n_epochs'] = 50  # Paper says "up to 200", but 50 is reasonable for demo/CPU
    config['training']['patience'] = 15   # Early stopping patience
    config['training']['batch_size'] = 256
    config['training']['learning_rate'] = 0.001
    config['model']['hidden_dim'] = 128
    config['model']['layers'] = 4
    config['model']['dropout'] = 0.2
    
    # Use a moderate dataset size to balance fidelity and runtime
    # Paper uses 100k/degree -> 300k total. Here we use ~10% of that for speed.
    config['data']['train_samples'] = 24000
    config['data']['val_samples'] = 3000
    config['data']['test_samples'] = 3000
    
    config['training']['model_save_path'] = './models/rank_gnn_paper.pth'
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(config['training']['model_save_path']), exist_ok=True)

    print(f"Configuration set: Epochs={config['training']['n_epochs']}, Batch={config['training']['batch_size']}, LR={config['training']['learning_rate']}")
    print(f"Dataset: Train={config['data']['train_samples']}, Val={config['data']['val_samples']}, Test={config['data']['test_samples']}")

    print("\n--- Starting Training ---")
    start_time = time.time()
    try:
        model = train_model(config)
        print(f"Training completed successfully in {time.time() - start_time:.2f}s.")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Starting Evaluation ---")
    try:
        # Load the best model
        model.load_state_dict(torch.load(config['training']['model_save_path']))
        
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
