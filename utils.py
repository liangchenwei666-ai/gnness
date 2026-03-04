import numpy as np

import torch

import os

import yaml



def load_config(config_path):

    with open(config_path, 'r') as f:

        config = yaml.safe_load(f)

    return config



def save_config(config, config_path):

    with open(config_path, 'w') as f:

        yaml.dump(config, f, default_flow_style=False)



def load_model(model_class, model_path, **model_kwargs):

    model = model_class(**model_kwargs)

    model.load_state_dict(torch.load(model_path))

    model.eval()

    return model



def save_model(model, model_path):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(model.state_dict(), model_path)



def validate_coefficients(a):

    if not isinstance(a, np.ndarray):

        a = np.array(a)

    if len(a) == 0:

        raise ValueError("Coefficients array is empty")

    if np.isnan(a).any() or np.isinf(a).any():

        raise ValueError("Coefficients contain NaN or Inf values")

    return a



def normalize_coefficients(a):

    a = validate_coefficients(a)

    norm = np.linalg.norm(a)

    if norm > 0:

        return a / norm

    else:

        return a



def compute_stability_score(a, roots, weights):

    residual = np.sqrt(np.sum((a - np.sum(weights[:, None] * (roots[:, None] ** np.arange(len(a))), axis=0)) ** 2))

    condition_number = np.linalg.cond(np.vander(roots, len(roots), increasing=True))

    stability = 1.0 / (1.0 + residual * condition_number)

    return stability



def visualize_coefficients(a, save_path=None):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.plot(range(len(a)), a, 'o-')

    plt.title('Coefficients')

    plt.xlabel('Index')

    plt.ylabel('Value')

    plt.grid(True)

    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path)

    else:

        plt.show()

    plt.close()



def visualize_roots(roots, save_path=None):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))

    plt.scatter(np.real(roots), np.imag(roots), s=100)

    plt.title('Roots')

    plt.xlabel('Real Part')

    plt.ylabel('Imaginary Part')

    plt.grid(True)

    plt.axis('equal')

    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path)

    else:

        plt.show()

    plt.close()



def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):

    n = len(dataset)

    train_size = int(n * train_ratio)

    val_size = int(n * val_ratio)

    test_size = n - train_size - val_size

    train_dataset = dataset[:train_size]

    val_dataset = dataset[train_size:train_size+val_size]

    test_dataset = dataset[train_size+val_size:]

    return train_dataset, val_dataset, test_dataset



def generate_batch_indices(n, batch_size):

    indices = np.arange(n)

    np.random.shuffle(indices)

    batches = []

    for i in range(0, n, batch_size):

        batch = indices[i:i+batch_size]

        batches.append(batch)

    return batches



def print_metrics(metrics, method_name):

    print(f"\n{method_name} Metrics:")

    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    print(f"  VSR: {metrics['vsr']:.4f}")

    print(f"  Avg Runtime: {metrics['avg_runtime']:.4f}s")

    print(f"  Std Runtime: {metrics['std_runtime']:.4f}s")

    print(f"  Avg Residual: {metrics['avg_residual']:.4f}")

    print(f"  Std Residual: {metrics['std_residual']:.4f}")
