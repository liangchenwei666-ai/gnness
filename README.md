# GNNESS: GNN-guided Solver

This repository contains the implementation of GNNESS, a Graph Neural Network guided solver.

## Directory Structure

- `models/`: Pre-trained model weights.
- `graph_builder.py`: Logic for building graph representations from input data.
- `solver.py`: Core solver implementation.
- `train.py`: Training script for the GNN model.
- `evaluate.py`: Evaluation scripts.
- `run_*.py`: Various entry points for running experiments and benchmarks.

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following packages:
- `torch`
- `torch_geometric`
- `numpy`
- `scipy`
- `pyyaml`
- `tqdm`
- `matplotlib`

### Running the Code

To run the fast version of GNNESS:
```bash
python run_fast_gnness.py
```

To reproduce paper experiments:
```bash
python run_paper_experiments.py
```

## Configuration

The main configuration file is `config.yaml`. You can modify parameters there to adjust the experiment settings.

## Notes

The `backup/` directory contains legacy files and previous directory structures which are not part of the main codebase.
