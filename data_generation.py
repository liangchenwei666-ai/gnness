import numpy as np
import random
from tqdm import tqdm

def sample_distinct_roots(r, distribution='uniform', min_spacing=1e-2, max_spacing=None):
    roots = []
    while len(roots) < r:
        if distribution == 'uniform':
            root = np.random.uniform(-1, 1)
        elif distribution == 'normal':
            root = np.random.normal(0, 1)
        else:
            raise ValueError("Unsupported distribution")
        spacing_valid = all(abs(root - existing) > min_spacing for existing in roots)
        if max_spacing is not None:
            spacing_valid = spacing_valid and all(abs(root - existing) < max_spacing for existing in roots)
        if spacing_valid:
            roots.append(root)
    return np.array(roots)

def sample_weights(r, min_magnitude=1e-2, distribution='normal'):
    weights = []
    while len(weights) < r:
        if distribution == 'normal':
            weight = np.random.normal(0, 1)
        elif distribution == 'lognormal':
            weight = np.random.lognormal(0, 1) * random.choice([1, -1])
        else:
            raise ValueError("Unsupported distribution")
        if abs(weight) >= min_magnitude:
            weights.append(weight)
    weights = np.array(weights)
    weights /= np.linalg.norm(weights)
    return weights

def generate_sample(d, r, noise_level=0.0, root_distribution='uniform', min_spacing=1e-2, max_spacing=None, min_weight_magnitude=1e-2, weight_distribution='normal'):
    noise_level = float(noise_level)
    # Use min_spacing=1e-2 as recommended in the paper to avoid near-collision roots
    roots = sample_distinct_roots(r, distribution=root_distribution, min_spacing=min_spacing, max_spacing=max_spacing)
    # Sample weights from normal distribution and ensure minimum magnitude
    weights = sample_weights(r, min_magnitude=min_weight_magnitude, distribution=weight_distribution)
    # Generate coefficients using the moment identity: a_i = sum(weights * roots^i)
    coefficients = []
    for i in range(d + 1):
        a_i = np.sum(weights * (roots ** i))
        coefficients.append(a_i)
    coefficients = np.array(coefficients)
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(coefficients))
        coefficients += noise
    raw_coeff = coefficients.copy()
    # Scale normalization
    eps = 1e-12
    norm = np.linalg.norm(coefficients) + eps
    normalized_coeff = coefficients / norm
    return normalized_coeff, raw_coeff, r, roots, weights

def generate_near_collision_sample(d, r, noise_level=0.0, min_spacing=1e-4, max_spacing=1e-3):
    noise_level = float(noise_level)
    roots = sample_distinct_roots(r, distribution='uniform', min_spacing=min_spacing, max_spacing=max_spacing)
    weights = sample_weights(r, min_magnitude=1e-2)
    coefficients = []
    for i in range(d + 1):
        a_i = np.sum(weights * (roots ** i))
        coefficients.append(a_i)
    coefficients = np.array(coefficients)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(coefficients))
        coefficients += noise
    raw_coeff = coefficients.copy()
    eps = 1e-12
    norm = np.linalg.norm(coefficients) + eps
    normalized_coeff = coefficients / norm
    return normalized_coeff, raw_coeff, r, roots, weights

def generate_ill_conditioned_sample(d, r, noise_level=0.0):
    noise_level = float(noise_level)
    roots = sample_distinct_roots(r, distribution='uniform')
    weights = sample_weights(r, min_magnitude=1e-2, distribution='lognormal')
    coefficients = []
    for i in range(d + 1):
        a_i = np.sum(weights * (roots ** i))
        coefficients.append(a_i)
    coefficients = np.array(coefficients)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(coefficients))
        coefficients += noise
    raw_coeff = coefficients.copy()
    eps = 1e-12
    norm = np.linalg.norm(coefficients) + eps
    normalized_coeff = coefficients / norm
    return normalized_coeff, raw_coeff, r, roots, weights

import torch

def generate_sample_gpu(d, r, noise_level=0.0, root_distribution='uniform', min_spacing=1e-2, max_spacing=None, min_weight_magnitude=1e-2, weight_distribution='normal', device='cuda'):
    # Sample roots on GPU
    # Note: Rejection sampling for spacing is hard to vectorize efficiently on GPU for small r
    # So we generate candidates on CPU or use a simple heuristic, but let's try to stick to CPU for sampling params
    # and use GPU for the heavy lifting: coefficient generation
    
    # Generate roots and weights on CPU (logic is complex but light)
    roots_cpu = sample_distinct_roots(r, distribution=root_distribution, min_spacing=min_spacing, max_spacing=max_spacing)
    weights_cpu = sample_weights(r, min_magnitude=min_weight_magnitude, distribution=weight_distribution)
    
    # Move to GPU for coefficient computation
    roots = torch.tensor(roots_cpu, device=device, dtype=torch.float64) # Use double precision
    weights = torch.tensor(weights_cpu, device=device, dtype=torch.float64)
    
    # a_i = sum(weights * roots^i)
    # Vectorized: a = weights @ (roots ^ [0..d])
    indices = torch.arange(d + 1, device=device, dtype=torch.float64).unsqueeze(0) # [1, d+1]
    roots_expanded = roots.unsqueeze(1) # [r, 1]
    powers = torch.pow(roots_expanded, indices) # [r, d+1]
    
    # Sum over r (dim 0) weighted by weights
    coefficients = torch.sum(weights.unsqueeze(1) * powers, dim=0) # [d+1]
    
    # Add noise
    noise_level = float(noise_level)
    if noise_level > 0:
        noise = torch.randn(d + 1, device=device, dtype=torch.float64) * noise_level
        coefficients += noise
        
    raw_coeff = coefficients.clone()
    
    # Normalization
    eps = 1e-12
    norm = torch.norm(coefficients) + eps
    normalized_coeff = coefficients / norm
    
    # Move back to CPU numpy for compatibility with existing pipeline
    return normalized_coeff.cpu().numpy(), raw_coeff.cpu().numpy(), r, roots_cpu, weights_cpu

def generate_dataset(n_samples, d_range, r_range, noise_levels=[0.0, 1e-6, 1e-4, 1e-3, 1e-2], include_hard_samples=False):
    dataset = []
    # Calculate samples per degree to ensure balanced distribution
    samples_per_degree = n_samples // len(d_range)
    remaining_samples = n_samples % len(d_range)
    
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    if use_gpu:
        print(f"Accelerating data generation with {device}")
    
    # Generate samples for each degree to ensure exactly the right number per degree
    current_sample = 0
    for d in d_range:
        # Calculate number of samples for this degree
        degree_samples = samples_per_degree
        if remaining_samples > 0:
            degree_samples += 1
            remaining_samples -= 1
        
        # Generate samples for this degree
        for i in tqdm(range(degree_samples), desc=f"Generating samples for degree {d}"):
            # Sample r uniformly from {1, 2, ..., 10} for balanced classification
            r = random.randint(r_range[0], r_range[1])
            noise_level = random.choice(noise_levels)
            
            is_hard = False
            sample_type = 'standard'
            
            if include_hard_samples and random.random() < 0.3:
                is_hard = True
                if random.random() < 0.5:
                    sample_type = 'near_collision'
                    # Use CPU generator for complex logic of hard samples
                    a, raw_a, true_rank, roots, weights = generate_near_collision_sample(d, r, noise_level)
                else:
                    sample_type = 'ill_conditioned'
                    a, raw_a, true_rank, roots, weights = generate_ill_conditioned_sample(d, r, noise_level)
            
            if not is_hard:
                if use_gpu:
                     a, raw_a, true_rank, roots, weights = generate_sample_gpu(d, r, noise_level, device=device)
                else:
                    a, raw_a, true_rank, roots, weights = generate_sample(d, r, noise_level)
            
            dataset.append({
            'a': a,
            'raw_a': raw_a,
            'true_rank': true_rank,
            'roots': roots,
            'weights': weights,
            'd': d,
            'noise_level': noise_level,
            'sample_type': sample_type
        })
    return dataset

def generate_dataset_by_degree(degrees, samples_per_degree, r_range, noise_levels=[0.0, 1e-6, 1e-4, 1e-3, 1e-2], include_hard_samples=False):
    dataset = []
    for d in degrees:
        for i in range(samples_per_degree):
            r = random.randint(r_range[0], r_range[1])
            noise_level = random.choice(noise_levels)
            if include_hard_samples and random.random() < 0.3:
                if random.random() < 0.5:
                    a, raw_a, true_rank, roots, weights = generate_near_collision_sample(d, r, noise_level)
                    sample_type = 'near_collision'
                else:
                    a, raw_a, true_rank, roots, weights = generate_ill_conditioned_sample(d, r, noise_level)
                    sample_type = 'ill_conditioned'
            else:
                a, raw_a, true_rank, roots, weights = generate_sample(d, r, noise_level)
                sample_type = 'standard'
            dataset.append({
                'a': a,
                'raw_a': raw_a,
                'true_rank': true_rank,
                'roots': roots,
                'weights': weights,
                'd': d,
                'noise_level': noise_level,
                'sample_type': sample_type
            })
    return dataset

def split_dataset_with_balance(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    by_rank = {}
    for sample in dataset:
        r = sample['true_rank']
        if r not in by_rank:
            by_rank[r] = []
        by_rank[r].append(sample)
    
    train_set = []
    val_set = []
    test_set = []
    
    for r in by_rank:
        samples = by_rank[r]
        np.random.shuffle(samples)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        
        train_set.extend(samples[:n_train])
        val_set.extend(samples[n_train:n_train+n_val])
        test_set.extend(samples[n_train+n_val:])
    
    np.random.shuffle(train_set)
    np.random.shuffle(val_set)
    np.random.shuffle(test_set)
    
    return train_set, val_set, test_set