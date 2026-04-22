

import os
import math
import json
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SEED         = 42
DATA_DIR     = "./data"
RESULTS_DIR  = "./results"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

LAMBDA_VALUES = [0.0, 0.001, 0.01]

BATCH_SIZE    = 128
EPOCHS        = 30
LEARNING_RATE = 1e-3
GATE_LR       = 1e-2
WEIGHT_DECAY  = 1e-4

GATE_THRESHOLD = 1e-2

def set_seed(seed: int = SEED) -> None:
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_features
        bound  = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        nn.init.uniform_(self.gate_scores, -0.5, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        gates          = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gate_values(self) -> torch.Tensor:
        
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def get_sparsity(self, threshold: float = GATE_THRESHOLD) -> float:
        
        gates = self.get_gate_values()
        return (gates < threshold).sum().item() / gates.numel()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}")

class SelfPruningNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def compute_sparsity_loss(self) -> torch.Tensor:
        
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                loss = loss + torch.sigmoid(m.gate_scores).sum()
        return loss

    @torch.no_grad()
    def get_layer_sparsities(self,
                             threshold: float = GATE_THRESHOLD) -> Dict[str, float]:
        
        out: Dict[str, float] = {}
        for name, m in self.named_modules():
            if isinstance(m, PrunableLinear):
                out[name] = m.get_sparsity(threshold) * 100.0
        return out

    @torch.no_grad()
    def get_overall_sparsity(self,
                             threshold: float = GATE_THRESHOLD) -> float:
        
        pruned = total = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                g = m.get_gate_values()
                pruned += (g < threshold).sum().item()
                total  += g.numel()
        return (pruned / total) * 100.0 if total else 0.0

    @torch.no_grad()
    def get_all_gate_values(self) -> np.ndarray:
        
        parts = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                parts.append(m.get_gate_values().cpu().numpy().ravel())
        return np.concatenate(parts)

    def count_parameters(self) -> Dict[str, int]:
        total    = sum(p.numel() for p in self.parameters())
        weights  = sum(m.weight.numel()
                       for m in self.modules() if isinstance(m, PrunableLinear))
        gates    = sum(m.gate_scores.numel()
                       for m in self.modules() if isinstance(m, PrunableLinear))
        return {
            "total":      total,
            "weights":    weights,
            "gates":      gates,
            "other":      total - weights - gates,
        }

def get_data_loaders(
    batch_size: int = BATCH_SIZE,
    data_dir: str   = DATA_DIR,
) -> Tuple[DataLoader, DataLoader]:
    
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    return train_loader, test_loader

def train_one_epoch(
    model: SelfPruningNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_s: float,
    device: torch.device,
) -> Dict[str, float]:
    
    model.train()
    cum_cls = cum_sp = cum_total = 0.0
    correct = total = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs  = model(inputs)
        cls_loss = F.cross_entropy(outputs, targets)
        sp_loss  = model.compute_sparsity_loss()
        loss     = cls_loss + lambda_s * sp_loss

        loss.backward()
        optimizer.step()

        cum_cls   += cls_loss.item()
        cum_sp    += sp_loss.item()
        cum_total += loss.item()

        _, preds = outputs.max(dim=1)
        total   += targets.size(0)
        correct += preds.eq(targets).sum().item()

    n = len(loader)
    return {
        "cls_loss":     cum_cls   / n,
        "sparsity_loss": cum_sp  / n,
        "total_loss":   cum_total / n,
        "accuracy":     100.0 * correct / total,
        "sparsity_pct": model.get_overall_sparsity(),
    }

@torch.no_grad()
def evaluate(
    model: SelfPruningNetwork,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    
    model.eval()
    cum_loss = 0.0
    correct  = total = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        cum_loss += F.cross_entropy(outputs, targets).item()

        _, preds = outputs.max(dim=1)
        total   += targets.size(0)
        correct += preds.eq(targets).sum().item()

    return {
        "test_loss":     cum_loss / len(loader),
        "test_accuracy": 100.0 * correct / total,
        "sparsity_pct":  model.get_overall_sparsity(),
    }

def run_experiment(
    lambda_s: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
) -> Dict:
    
    banner = f"  EXPERIMENT · λ = {lambda_s:.1e}"
    print(f"\n{'═' * 70}")
    print(banner)
    print(f"{'═' * 70}")

    set_seed()
    model = SelfPruningNetwork().to(device)

    info = model.count_parameters()
    print(f"  Parameters — total: {info['total']:,}  "
          f"weights: {info['weights']:,}  "
          f"gates: {info['gates']:,}  other: {info['other']:,}")
    print(f"  Device: {device}")

    weights_biases = []
    gate_params = []
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            weights_biases.append(param)

    optimizer = optim.Adam([
        {"params": weights_biases, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        {"params": gate_params,    "lr": GATE_LR,       "weight_decay": 0.0}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_acc": [], "test_acc": [],
        "cls_loss": [], "sp_loss": [], "total_loss": [],
        "sparsity": [], "lr": [],
    }
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr  = train_one_epoch(model, train_loader, optimizer, lambda_s, device)
        te  = evaluate(model, test_loader, device)
        scheduler.step()
        lr  = scheduler.get_last_lr()[0]

        history["train_acc"].append(tr["accuracy"])
        history["test_acc"].append(te["test_accuracy"])
        history["cls_loss"].append(tr["cls_loss"])
        history["sp_loss"].append(tr["sparsity_loss"])
        history["total_loss"].append(tr["total_loss"])
        history["sparsity"].append(te["sparsity_pct"])
        history["lr"].append(lr)

        if te["test_accuracy"] > best_acc:
            best_acc = te["test_accuracy"]

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{epochs} │ "
                f"Train {tr['accuracy']:5.1f}% │ "
                f"Test {te['test_accuracy']:5.1f}% │ "
                f"CLS {tr['cls_loss']:.4f} │ "
                f"Sparsity {te['sparsity_pct']:5.1f}% │ "
                f"LR {lr:.6f} │ "
                f"{elapsed:.0f}s"
            )

    total_time = time.time() - t0
    final  = evaluate(model, test_loader, device)
    layers = model.get_layer_sparsities()
    gates  = model.get_all_gate_values()

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  RESULTS   λ = {lambda_s:.1e}")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  Test Accuracy     {final['test_accuracy']:>7.2f} %")
    print(f"  │  Best Test Acc     {best_acc:>7.2f} %")
    print(f"  │  Overall Sparsity  {final['sparsity_pct']:>7.2f} %")
    print(f"  │  Training Time     {total_time:>7.1f} s")
    print(f"  ├──────────────────────────────────────────────┤")
    for ln, sp in layers.items():
        print(f"  │  {ln:<22s} {sp:>7.2f} %")
    print(f"  └──────────────────────────────────────────────┘")

    return {
        "lambda":             lambda_s,
        "test_accuracy":      final["test_accuracy"],
        "best_test_accuracy": best_acc,
        "sparsity_pct":       final["sparsity_pct"],
        "layer_sparsities":   layers,
        "time_s":             total_time,
        "history":            history,
        "gate_values":        gates,
        "model":              model,
    }

PALETTE = ["#2196F3", "#FF9800", "#E91E63"]

def _style_axis(ax: plt.Axes) -> None:
    
    ax.grid(axis="both", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_gate_distribution(results: List[Dict], out: str) -> str:
    
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res, clr in zip(axes, results, PALETTE):
        g   = res["gate_values"]
        lam = res["lambda"]
        sp  = res["sparsity_pct"]

        ax.hist(g, bins=100, color=clr, edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.axvline(GATE_THRESHOLD, color="#E74C3C", ls="--", lw=1.5,
                   label=f"Threshold = {GATE_THRESHOLD}")
        ax.set_title(f"λ = {lam:.1e}\nSparsity {sp:.1f} %",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Count")
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=9)
        _style_axis(ax)

    fig.suptitle("Distribution of Gate Values After Training",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(out, "gate_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved → {path}")
    return path

def plot_training_curves(results: List[Dict], out: str) -> str:
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, res in enumerate(results):
        h   = res["history"]
        lam = res["lambda"]
        ep  = range(1, len(h["train_acc"]) + 1)
        c   = PALETTE[i]
        lab = f"λ = {lam:.1e}"

        axes[0, 0].plot(ep, h["test_acc"], color=c, lw=2, label=lab)
        axes[0, 1].plot(ep, h["cls_loss"], color=c, lw=2, label=lab)
        axes[1, 0].plot(ep, h["sparsity"], color=c, lw=2, label=lab)
        axes[1, 1].plot(ep, h["train_acc"], color=c, lw=2, ls="--", alpha=0.5)
        axes[1, 1].plot(ep, h["test_acc"],  color=c, lw=2, label=lab)

    titles = ["Test Accuracy (%)", "Classification Loss (CE)",
              "Network Sparsity (%)", "Train ╌╌ vs Test ── Accuracy (%)"]
    ylabels = ["Accuracy (%)", "Loss", "Sparsity (%)", "Accuracy (%)"]
    for ax, t, yl in zip(axes.flat, titles, ylabels):
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_ylabel(yl)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        _style_axis(ax)

    fig.suptitle("Training Dynamics Across λ Values",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(out, "training_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 Saved → {path}")
    return path

def plot_layer_sparsity(results: List[Dict], out: str) -> str:
    
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = list(results[0]["layer_sparsities"].keys())
    x      = np.arange(len(layers))
    w      = 0.25

    for i, res in enumerate(results):
        vals   = [res["layer_sparsities"][l] for l in layers]
        offset = (i - len(results) / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w, label=f"λ = {res['lambda']:.1e}",
                        color=PALETTE[i], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if v > 2:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Sparsity (%)", fontsize=12)
    ax.set_title("Layer-wise Sparsity Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(".", "\n") for n in layers], fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    _style_axis(ax)

    plt.tight_layout()
    path = os.path.join(out, "layer_sparsity_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved → {path}")
    return path

def plot_sparsity_accuracy_tradeoff(results: List[Dict], out: str) -> str:
    
    fig, ax = plt.subplots(figsize=(8, 5))

    sparsities = [r["sparsity_pct"] for r in results]
    accuracies = [r["test_accuracy"] for r in results]
    lambdas    = [r["lambda"]        for r in results]

    ax.plot(sparsities, accuracies, "o-", color="#4A90D9", lw=2, ms=10,
            markeredgecolor="white", markeredgewidth=2)

    for s, a, lam in zip(sparsities, accuracies, lambdas):
        ax.annotate(f"λ={lam:.0e}", (s, a), textcoords="offset points",
                    xytext=(10, 8), fontsize=9, fontweight="bold")

    ax.set_xlabel("Sparsity (%)",   fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Sparsity–Accuracy Trade-off",
                 fontsize=14, fontweight="bold")
    _style_axis(ax)

    plt.tight_layout()
    path = os.path.join(out, "sparsity_accuracy_tradeoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 Saved → {path}")
    return path

def main() -> List[Dict]:
    header = (
        "\n"
        "═" * 70 + "\n"
        "  SELF-PRUNING NEURAL NETWORK FOR CIFAR-10\n"
        "  Tredence Analytics — AI Engineering Case Study\n"
        "═" * 70
    )
    print(header)
    print(f"  Device          : {DEVICE}")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Weight decay    : {WEIGHT_DECAY}")
    print(f"  Lambda values   : {LAMBDA_VALUES}")
    print(f"  Gate threshold  : {GATE_THRESHOLD}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n  Loading CIFAR-10 …")
    train_loader, test_loader = get_data_loaders()
    print(f"  Train: {len(train_loader.dataset):,} samples  "
          f"Test: {len(test_loader.dataset):,} samples")

    all_results: List[Dict] = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(lam, train_loader, test_loader, DEVICE, EPOCHS)
        all_results.append(result)

    print(f"\n\n{'═' * 70}")
    print("  COMPARATIVE SUMMARY")
    print(f"{'═' * 70}\n")
    print(f"  {'Lambda':<12} {'Test Acc':>10} {'Sparsity':>10} {'Time':>10}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10}")
    for r in all_results:
        print(f"  {r['lambda']:<12.1e} "
              f"{r['test_accuracy']:>9.2f}% "
              f"{r['sparsity_pct']:>9.2f}% "
              f"{r['time_s']:>9.1f}s")

    print("\n  Generating visualisations …")
    plot_gate_distribution(all_results, RESULTS_DIR)
    plot_training_curves(all_results, RESULTS_DIR)
    plot_layer_sparsity(all_results, RESULTS_DIR)
    plot_sparsity_accuracy_tradeoff(all_results, RESULTS_DIR)

    serializable = []
    for r in all_results:
        serializable.append({
            "lambda":             r["lambda"],
            "test_accuracy":      r["test_accuracy"],
            "best_test_accuracy": r["best_test_accuracy"],
            "sparsity_pct":       r["sparsity_pct"],
            "layer_sparsities":   r["layer_sparsities"],
            "time_s":             r["time_s"],
        })
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  💾 Results JSON → {json_path}")

    print(f"\n{'═' * 70}")
    print("  ✅  ALL EXPERIMENTS COMPLETE")
    print(f"{'═' * 70}\n")

    return all_results

if __name__ == "__main__":
    main()
