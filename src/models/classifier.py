"""
Training and evaluation loops for the respiratory sound classifier.

Supports:
- Standard training (clean data only)
- Noise-aware training (on-the-fly SNR injection during training)
- Evaluation with per-class and macro-averaged metrics
- Early stopping with patience
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    noise_fn=None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training data loader yielding (waveform, label) pairs.
        criterion: Loss function (e.g., FocalLoss).
        optimizer: Optimizer (e.g., Adam).
        device: Device to train on (cpu/cuda).
        noise_fn: Optional callable(waveform) -> noisy_waveform for noise-aware training.
                  If provided, applies random noise injection to each batch on the fly.

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Apply noise augmentation if provided (noise-aware training)
        if noise_fn is not None:
            batch_x = noise_fn(batch_x)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model: The neural network model.
        dataloader: Evaluation data loader.
        criterion: Loss function.
        device: Device.
        num_classes: Number of classes.

    Returns:
        Dictionary with keys: loss, accuracy, balanced_accuracy, f1_macro,
        precision_macro, recall_macro, and per-class f1 scores.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision_macro": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
    }

    # Per-class F1
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, zero_division=0, labels=range(num_classes)
    )
    for i, f1_val in enumerate(per_class_f1):
        results[f"f1_class_{i}"] = float(f1_val)

    return results


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.

    Monitors a metric (lower is better for 'loss', higher is better for others)
    and stops after `patience` epochs without improvement.
    """

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs to wait before stopping.
            mode: 'min' if monitoring loss (lower=better), 'max' for metrics like F1.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    patience: int = 10,
    noise_fn=None,
    num_classes: int = 4,
    scheduler=None,
) -> Tuple[nn.Module, List[Dict]]:
    """
    Full training loop with early stopping and validation monitoring.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        noise_fn: Optional noise injection function for noise-aware training.
        num_classes: Number of classes.
        scheduler: Optional LR scheduler. If provided, scheduler.step() is called
                   after each epoch.

    Returns:
        Tuple of (best_model, history) where history is a list of dicts
        with per-epoch train/val metrics.
    """
    early_stopping = EarlyStopping(patience=patience, mode="max")
    best_model_state = None
    best_f1 = -1.0
    history = []

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, noise_fn
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_log)

        # Track best model by macro F1
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Early stopping check
        if early_stopping(val_metrics["f1_macro"]):
            break

        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
