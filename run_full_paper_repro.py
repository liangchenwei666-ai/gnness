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

    # Modify config to match paper's FULL training parameters
    print("Setting FULL training parameters according to paper...")
    
    # Paper parameters
    # Dataset size: 100,000 samples per degree * 3 degrees = 300,000 total
    # Split: 80/10/10 -> Train 240k, Val 30k, Test 30k
    config['data']['train_samples'] = 240000
    config['data']['val_samples'] = 30000
    config['data']['test_samples'] = 30000
    
    config['training']['n_epochs'] = 200  # Paper: "up to 200"
    config['training']['patience'] = 20   # Increased patience for full run
    config['training']['batch_size'] = 256
    config['training']['learning_rate'] = 0.001
    
    config['training']['model_save_path'] = './models/rank_gnn_full_paper.pth'
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(config['training']['model_save_path']), exist_ok=True)
    
    # Allow command line override for epochs (for testing)
    if len(sys.argv) > 1:
        try:
            config['training']['n_epochs'] = int(sys.argv[1])
            print(f"Overriding n_epochs to {config['training']['n_epochs']} from command line.")
        except ValueError:
            pass

    print(f"Configuration set: Epochs={config['training']['n_epochs']}, Batch={config['training']['batch_size']}")
    print(f"Dataset: Train={config['data']['train_samples']}, Val={config['data']['val_samples']}, Test={config['data']['test_samples']}")
    print("WARNING: This full-scale run involves generating 300,000 graphs and training for up to 200 epochs.")
    print("On a standard CPU, data generation may take ~30-60 mins, and training may take several hours.")
    
    print("\n--- Starting Training ---")
    start_time = time.time()
    try:
        model = train_model(config)
        print(f"Training completed successfully in {(time.time() - start_time)/3600:.2f} hours.")
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
