"""
Tests for the baseline CNN model and classifier training/evaluation utilities.
Uses synthetic data only.
"""

import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader
from src.models.baseline_cnn import BaselineCNN, FocalLoss, AttentionPooling
from src.models.classifier import (
    train_one_epoch,
    evaluate,
    EarlyStopping,
    train_model,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


class TestBaselineCNN:
    """Tests for the BaselineCNN architecture."""

    def test_forward_shape(self):
        """Output shape is (B, num_classes) for a valid input."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        # time_frames must be divisible by 16 (4 max-pool with stride 2)
        x = torch.randn(2, 1, 128, 160)
        out = model(x)
        assert out.shape == (2, 4)

    def test_forward_different_batch_sizes(self):
        """Works with batch sizes 1, 4, 16."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 1, 128, 160)
            out = model(x)
            assert out.shape == (bs, 4)

    def test_output_not_nan(self):
        """Output contains no NaN values."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        x = torch.randn(4, 1, 128, 160)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_get_feature_maps_shape(self):
        """Feature maps from the last conv block have expected shape."""
        model = BaselineCNN(num_classes=4, channels=(32, 64, 128, 256), n_mels=128)
        x = torch.randn(2, 1, 128, 160)
        fmaps = model.get_feature_maps(x)
        # After 4 max-pools: 128/16=8 freq, 160/16=10 time
        assert fmaps.shape == (2, 256, 8, 10)

    def test_gradient_flows(self):
        """Gradients flow through the model (no dead layers)."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        x = torch.randn(2, 1, 128, 160, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_different_time_lengths(self):
        """Model handles different time frame lengths (as long as divisible by 16)."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        for t in [32, 64, 128, 256]:
            x = torch.randn(2, 1, 128, t)
            out = model(x)
            assert out.shape == (2, 4)


class TestFocalLoss:
    """Tests for the FocalLoss implementation."""

    def test_loss_is_scalar(self):
        """Loss output is a scalar."""
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 4)
        targets = torch.tensor([0, 1, 2, 3])
        loss = criterion(logits, targets)
        assert loss.dim() == 0

    def test_loss_positive(self):
        """Loss is positive."""
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = criterion(logits, targets)
        assert loss.item() > 0

    def test_gamma_zero_matches_ce(self):
        """With gamma=0 and no alpha, focal loss should approximate cross-entropy."""
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))

        focal = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()

        focal_loss = focal(logits, targets)
        ce_loss = ce(logits, targets)

        assert torch.isclose(focal_loss, ce_loss, atol=1e-5)

    def test_higher_gamma_lower_loss_for_easy(self):
        """Higher gamma reduces loss for well-classified (easy) samples."""
        # Use moderately confident predictions so the gamma effect is measurable
        logits = torch.tensor([[3.0, -1.0, -1.0, -1.0]] * 8)
        targets = torch.zeros(8, dtype=torch.long)

        loss_g0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_g2 = FocalLoss(gamma=2.0)(logits, targets)

        # Higher gamma should reduce loss for easy examples
        assert loss_g2 < loss_g0

    def test_with_class_weights(self):
        """Loss works with per-class alpha weights."""
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = criterion(logits, targets)
        assert loss.item() > 0

    def test_no_reduction(self):
        """With reduction='none', output has shape (B,)."""
        criterion = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = criterion(logits, targets)
        assert loss.shape == (8,)


class TestAttentionPooling:
    """Tests for the attention pooling layer."""

    def test_output_shape(self):
        """Output is (B, C*F) after pooling over time."""
        pool = AttentionPooling(in_features=256 * 8)
        x = torch.randn(2, 256, 8, 10)
        out = pool(x)
        assert out.shape == (2, 256 * 8)

    def test_attention_weights_sum_to_one(self):
        """Attention weights across time should sum to 1 (softmax)."""
        pool = AttentionPooling(in_features=64 * 4)
        x = torch.randn(2, 64, 4, 20)

        B, C, F, T = x.shape
        x_perm = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        weights = pool.attention(x_perm)
        weights = torch.softmax(weights, dim=1)

        sums = weights.sum(dim=1).squeeze()
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestEarlyStopping:
    """Tests for the EarlyStopping utility."""

    def test_no_stop_when_improving(self):
        """Should not stop when metric keeps improving."""
        es = EarlyStopping(patience=3, mode="max")
        for score in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert not es(score)

    def test_stops_after_patience(self):
        """Should stop after patience epochs without improvement."""
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)  # best
        es(0.4)  # worse
        es(0.4)  # worse
        stopped = es(0.4)  # worse -> patience exhausted
        assert stopped

    def test_min_mode(self):
        """In min mode, lower values are better."""
        es = EarlyStopping(patience=2, mode="min")
        es(1.0)  # best
        es(0.5)  # better
        es(0.6)  # worse
        stopped = es(0.7)  # worse -> stop
        assert stopped


class TestTrainEval:
    """Tests for training and evaluation loops using a tiny synthetic dataset."""

    def _make_tiny_loaders(self, n=32):
        """Create tiny train/val loaders with random spectrogram-like data."""
        # Shape: (n, 1, 128, 32) — small time frames for speed
        x = torch.randn(n, 1, 128, 32)
        y = torch.randint(0, 4, (n,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def test_train_one_epoch_returns_loss(self):
        """Training returns a positive loss value."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = self._make_tiny_loaders()

        loss = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_returns_metrics(self):
        """Evaluation returns a dict with all expected metric keys."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        criterion = FocalLoss(gamma=2.0)
        loader = self._make_tiny_loaders()

        metrics = evaluate(model, loader, criterion, torch.device("cpu"))

        expected_keys = [
            "loss", "accuracy", "balanced_accuracy", "f1_macro",
            "precision_macro", "recall_macro",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)

    def test_train_model_with_early_stopping(self):
        """Full training loop runs without errors and returns history."""
        model = BaselineCNN(num_classes=4, n_mels=128)
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = self._make_tiny_loaders()
        val_loader = self._make_tiny_loaders(n=16)

        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            torch.device("cpu"), num_epochs=3, patience=2
        )

        assert len(history) > 0
        assert "epoch" in history[0]
        assert "train_loss" in history[0]
        assert "val_f1_macro" in history[0]
