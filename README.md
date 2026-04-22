# Self-Pruning Neural Network (CIFAR-10)

This repository contains the solution for the Tredence Analytics AI Engineering case study. It implements a feed-forward neural network on CIFAR-10 that learns to prune its own weights during training using a custom gated linear layer and L1 regularization.

## Requirements
- Python 3.9+
- PyTorch
- Torchvision
- Matplotlib
- NumPy

## How to Run
To train the network and generate the trade-off plots, just run the main script:
```bash
python self_pruning_network.py
```
This will automatically download the CIFAR-10 dataset (if not already downloaded) and run three separate experiments for $\lambda \in \{0.0, 0.001, 0.01\}$. All generated plots and results are saved in the `results/` folder.

## Architecture & Design
The network is a standard MLP using a custom `PrunableLinear` layer instead of PyTorch's default `nn.Linear`.

### Custom PrunableLinear Layer
The layer maintains standard weights and biases, but adds a learnable parameter `gate_scores`. During the forward pass, these scores are passed through a Sigmoid to create gates between 0 and 1. The original weights are then multiplied by these gates.

### Enhancements
Because standard MLPs generally max out around 52% on CIFAR-10 due to a lack of spatial convolution, the following enhancements were added to raise the baseline capacity up to ~61.5%:
- **BatchNorm1d**: Added after the prunable layers to keep moving variances stable.
- **CosineAnnealingLR**: Used as a learning rate scheduler for smoother convergence over the 30 epochs.
- **Differential Learning Rates**: Used `1e-3` for general weights and `1e-2` for `gate_scores`. This helps balance the classification gradients against the sparse L1 penalty so that gates don't immediately zero out.

### Gate Initialization
Gates are initialized uniformly between `[-0.5, 0.5]`. This starts the sigmoid output near 0.5, allowing the classification loss a little time to push important gates toward 1.0 before the sparsity penalty pulls them to 0.0.

## Results
Please refer to `REPORT.md` for the analysis of the experiments and sparsity/accuracy tables.
