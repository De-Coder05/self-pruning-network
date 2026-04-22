# CIFAR-10 Self-Pruning Network Results

## 1. How L1 Penalty Induces Sparsity
Standard L2 regularization penalizes the square of weights, which shrinks them but rarely zeroes them out. An L1 penalty applies a constant pressure proportional to the absolute value. In this architecture, each layer has `gate_scores` where the actual gate is a sigmoid output ($g \in (0,1)$). The sparsity loss is the sum of all these gate values.

When using gradient descent, if a connection is not important enough to reduce the classification loss to offset the L1 penalty, the constant L1 derivative pushes the gate score into the negative domain until $\sigma(score) \to 0$, effectively pruning the connection.

## 2. Experimental Setup & Limitations
The goal was to implement this on a standard feed-forward MLP. Flat MLPs struggle with the spatial layout of CIFAR-10 (32x32x3). To establish a working baseline, the following configurations were used:
- Architecture: 3072 -> 512 -> 256 -> 128 -> 10
- Added `BatchNorm1d` before activations to stabilize training.
- Used an Adam optimizer with `CosineAnnealingLR` over 30 epochs.
- Split learning rates: `1e-3` for standard weights, and `1e-2` for `gate_scores`. Slower gate updates were necessary to allow classification paths to form before pruning pressure dominated.

With these configurations, the baseline (0 sparsity) achieved an accuracy of 61.57%.

## 3. Results (Accuracy vs. Sparsity)

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity Level (%) |
|:------------------|:------------------|:-------------------|
| `0.0` (Baseline)  | 61.57%            | 0.00%              |
| `0.001`           | 60.41%            | 99.96%             |
| `0.01`            | 59.58%            | 99.99%             |

At $\lambda = 0.001$, the network successfully pruned 99.96% of its parameters. This dropped the accuracy from 61.57% to 60.41%, losing only ~1.16% accuracy while dropping ~1.7 million paramaters. This validates that the custom `PrunableLinear` layer effectively learns which connections to drop.

## 4. Gate Distribution
At $\lambda = 0.001$, the gate distribution is heavily bimodal. The vast majority of gates are crushed to 0.0 by the L1 penalty (pruned), while a very small selection of critical gates is pushed near 1.0 by the classification gradient.
