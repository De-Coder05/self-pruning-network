# Self-Pruning Neural Network for CIFAR-10

> **Case Study** — Tredence Analytics · AI Engineering Internship 2025  
> **Author** — Devansh Wadhwani

A PyTorch implementation of a feed-forward neural network that **learns to prune its own weights** during training. Each weight is paired with a learnable "gate" parameter; an L1 regularisation penalty drives unnecessary gates to zero, yielding a sparse, efficient network without any post-training pruning step.

---

## Table of Contents

1. [Architecture](#architecture)
2. [How It Works](#how-it-works)
3. [How to Run](#how-to-run)
4. [Project Structure](#project-structure)
5. [Design Decisions](#design-decisions)
6. [What Was Completed](#what-was-completed)
7. [What I Would Add With More Time](#what-i-would-add-with-more-time)

---

## Architecture

```
Input Image (32 × 32 × 3 = 3 072)
       │
  ┌────▼────────────────────────┐
  │  PrunableLinear(3072, 512)  │──► BatchNorm1d(512) ──► ReLU
  └─────────────────────────────┘
       │
  ┌────▼────────────────────────┐
  │  PrunableLinear(512, 256)   │──► BatchNorm1d(256) ──► ReLU
  └─────────────────────────────┘
       │
  ┌────▼────────────────────────┐
  │  PrunableLinear(256, 128)   │──► BatchNorm1d(128) ──► ReLU
  └─────────────────────────────┘
       │
  ┌────▼────────────────────────┐
  │  PrunableLinear(128, 10)    │──► Logits (Cross-Entropy)
  └─────────────────────────────┘
```

Every `PrunableLinear` layer stores **three** parameter tensors:

| Parameter      | Shape              | Purpose                                          |
|----------------|--------------------|--------------------------------------------------|
| `weight`       | `(out, in)`        | Standard weight matrix                           |
| `bias`         | `(out,)`           | Standard bias vector                             |
| `gate_scores`  | `(out, in)`        | Learnable scores → sigmoid → gates ∈ (0, 1)     |

---

## How It Works

### 1. Gated Forward Pass

```python
gates          = σ(gate_scores)        # ∈ (0, 1) — per-weight
pruned_weights = weight ⊙ gates        # element-wise
output         = x @ pruned_weights^T + bias
```

### 2. Sparsity Regularisation

```
Total Loss = CrossEntropy(ŷ, y) + λ · Σ σ(gate_scores)
                                       ↑
                              L1 norm of all gates
```

- Because gates are always ≥ 0 (sigmoid outputs), the L1 norm simplifies to a plain **sum**.
- Minimising this sum pushes gates toward **0**, pruning the associated weights.
- **λ** controls the sparsity–accuracy trade-off: higher λ → more pruning → lower accuracy.

### 3. Why L1 Encourages Sparsity

The L1 penalty applies a **constant-magnitude gradient** (±1) regardless of a parameter's current value. Even very small gate values receive the same "push" toward zero as large ones. In contrast, an L2 penalty applies a gradient proportional to magnitude, which *shrinks* values but rarely drives them to **exactly** zero.

Geometrically, the L1 constraint set forms a **diamond** (hypercube in high dimensions). Optimal solutions under L1 constraints naturally lie at the **corners** of this diamond — points where one or more coordinates are exactly zero.

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy
```

### Run the full experiment

```bash
cd self-pruning-network
python self_pruning_network.py
```

This will:
1. Download CIFAR-10 (first run only, ~170 MB)
2. Train three models with λ ∈ {1e-6, 5e-6, 2e-5}
3. Print per-epoch and final metrics to the terminal
4. Save all plots and results to `./results/`

### Output artefacts

| File                              | Description                                    |
|-----------------------------------|------------------------------------------------|
| `results/gate_distribution.png`        | Histogram of final gate values per λ      |
| `results/training_curves.png`          | Accuracy, loss, and sparsity over epochs  |
| `results/layer_sparsity_comparison.png`| Per-layer sparsity bar chart              |
| `results/sparsity_accuracy_tradeoff.png`| Pareto frontier: sparsity vs accuracy    |
| `results/experiment_results.json`      | Machine-readable results                  |

---

## Project Structure

```
self-pruning-network/
├── self_pruning_network.py   # Single-file solution (all components)
├── REPORT.md                 # Analysis report with results & plots
├── README.md                 # This file
├── data/                     # CIFAR-10 (auto-downloaded)
└── results/                  # Generated plots + JSON results
```

---

## Design Decisions

### 1. The custom `PrunableLinear` layer
A standard fully-connected layer is replaced with a custom layer wrapping both the original weight matrix ($W$) and a parallel set of `gate_scores` ($S$).
During the forward pass, $S$ passes through a Sigmoid activation, yielding $G \in (0, 1)$. The effective weight matrix is computed by $W \odot G$.

### 2. Out-of-Spec Architectural Enhancements (Crucial Disclosures)
The case study specifies a "standard feed-forward neural network." However, standard MLPs flattened against CIFAR-10's 3072 spatial inputs functionally cap out at ~52% accuracy (pure $nn.Linear$ testing). To create a viable testing ceiling for the gating mechanism, **the following components were deliberately added beyond standard specification**:
*   **BatchNorm1d**: Inserted between the PrunableLinear and ReLU to maintain feature statistics specifically as gates scale connections.
*   **CosineAnnealingLR**: Added for smooth loss convergence.
*   **Differential Learning Rates**: Used an `optim.Adam` parameter grouping to update `gate_scores` independently. 

These architectural improvements elevated the baseline MLP capacity to **61.5%**. Pruning percentages and trade-offs are strictly evaluated against *this enhanced model*, not against the standard `nn.Linear` layer alone.

### 3. Symmetric Initialization
The `gate_scores` are initialized uniformly between `[-0.5, 0.5]` rather than statically open. This provides the classification gradients a short "grace period" to organise structure and push important feature scores toward 1.0 before the sparse L1 penalty pulls weak elements into the negative domain.

| Decision | Rationale |
|----------|-----------|
| **Kaiming initialisation for weights** | Standard for ReLU networks (He et al., 2015) — ensures variance is preserved through forward and backward passes at initialisation. |
| **Three λ values (`0.0`, `0.001`, `0.01`)** | Chosen to span the spectrum from the true MLP baseline ceiling (0% pruning), through gentle pruning, and into aggressive capacity trade-offs. |

---

## What Was Completed

- [x] Custom `PrunableLinear` layer with learnable gate parameters
- [x] Correct gradient flow through both `weight` and `gate_scores`
- [x] L1 sparsity regularisation loss
- [x] Full training loop with composite loss
- [x] Evaluation on CIFAR-10 test set
- [x] Comparison across three λ values
- [x] Gate distribution histogram
- [x] Training curves (accuracy, loss, sparsity)
- [x] Layer-wise sparsity analysis
- [x] Sparsity–accuracy trade-off plot
- [x] Comprehensive Markdown report

---

## What I Would Add With More Time

- **Hard pruning at inference** — After training, permanently zero out pruned gates and restructure the weight matrices for genuine FLOPs reduction.
- **Convolutional backbone** — Replace the flatten + FC architecture with `PrunableConv2d` layers for significantly higher baseline accuracy.
- **Structured pruning** — Instead of individual weight gates, learn gates per neuron (row-level) to enable actual layer width reduction.
- **Dynamic λ scheduling** — Gradually increase λ during training (curriculum-style) for a smoother convergence trajectory.
- **Knowledge distillation** — Use the unpruned dense network as a teacher to recover accuracy lost to aggressive pruning.
- **ONNX export** — Export the pruned model for deployment benchmarking.

---

## References

- He, K. et al. (2015). *Delving Deep into Rectifiers.* ICCV.
- Louizos, C. et al. (2018). *Learning Sparse Neural Networks through L0 Regularization.* ICLR.
- Molchanov, P. et al. (2017). *Variational Dropout Sparsifies Deep Neural Networks.* ICML.
