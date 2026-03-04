import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data_generation import generate_dataset
from graph_builder import build_batch_graphs
from models import RankGNN
import yaml
import time
import os
from data_generation import split_dataset_with_balance

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config):
    # Generate training dataset
    train_dataset = generate_dataset(
        n_samples=config['data']['train_samples'],
        d_range=config['data']['d_range'],
        r_range=config['data']['r_range'],
        noise_levels=config['data']['noise_levels'],
        include_hard_samples=config['data']['include_hard_samples']
    )
    
    # Generate validation dataset
    val_dataset = generate_dataset(
        n_samples=config['data']['val_samples'],
        d_range=config['data']['d_range'],
        r_range=config['data']['r_range'],
        noise_levels=config['data']['noise_levels'],
        include_hard_samples=config['data']['include_hard_samples']
    )
    
    # Pre-calculate stats on training data
    train_graphs, train_stats = build_batch_graphs(
        train_dataset,
        R_max=config['model']['R_max'],
        delta=config['model']['delta'],
        normalize_features=True,
        stats=None
    )
    
    # Save training stats for later use during evaluation/inference
    torch.save(train_stats, 'train_stats.pt')
    
    val_graphs, _ = build_batch_graphs(
        val_dataset,
        R_max=config['model']['R_max'],
        delta=config['model']['delta'],
        normalize_features=True,
        stats=train_stats
    )
    train_loader = DataLoader(
        train_graphs,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    model = RankGNN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion_rank = nn.CrossEntropyLoss()
    criterion_stability = nn.BCELoss()
    best_val_loss = float('inf')
    patience = config['training']['patience']
    patience_counter = 0
    model_save_path = config['training']['model_save_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    for epoch in range(config['training']['n_epochs']):
        model.train()
        total_train_loss = 0
        start_time = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_weight = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            rank_pred, stability_pred = model(batch.x, batch.edge_index, batch.batch, edge_weight)
            loss_rank = criterion_rank(rank_pred, batch.y_rank)
            loss_stability = criterion_stability(stability_pred, batch.y_stability)
            loss = loss_rank + config['training']['lambda_aux'] * loss_stability
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_loss = 0
        total_val_rank_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_weight = batch.edge_attr if hasattr(batch, 'edge_attr') else None
                rank_pred, stability_pred = model(batch.x, batch.edge_index, batch.batch, edge_weight)
                loss_rank = criterion_rank(rank_pred, batch.y_rank)
                loss_stability = criterion_stability(stability_pred, batch.y_stability)
                loss = loss_rank + config['training']['lambda_aux'] * loss_stability
                total_val_loss += loss.item()
                
                # Calculate accuracy
                pred_labels = rank_pred.argmax(dim=1)
                acc = (pred_labels == batch.y_rank).sum().item() / batch.y_rank.size(0)
                total_val_rank_acc += acc
                
        val_loss = total_val_loss / len(val_loader)
        val_acc = total_val_rank_acc / len(val_loader)
        end_time = time.time()
        print(f'Epoch {epoch+1}/{config["training"]["n_epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Rank Acc: {val_acc:.4f}, Time: {end_time-start_time:.2f}s')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    return model

def main():
    config = load_config('config.yaml')
    model = train_model(config)
    print('Training completed!')

if __name__ == '__main__':
    main()