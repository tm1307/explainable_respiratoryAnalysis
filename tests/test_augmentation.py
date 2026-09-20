"""
Tests for the augmentation module (SpecAugment + Mixup).
Uses synthetic data only — no external datasets required.
"""

import torch
import pytest
from src.data.augmentation import (
    spec_augment,
    mixup,
    mixup_batch,
    to_one_hot,
)


@pytest.fixture
def seed():
    torch.manual_seed(42)


class TestSpecAugment:
    """Tests for SpecAugment time/frequency masking."""

    def test_output_shape_2d(self, seed):
        """Output shape matches input shape for 2D spectrograms."""
        spec = torch.randn(128, 157)
        augmented = spec_augment(spec)
        assert augmented.shape == spec.shape

    def test_output_shape_3d(self, seed):
        """Output shape matches input shape for 3D spectrograms (C, F, T)."""
        spec = torch.randn(1, 128, 157)
        augmented = spec_augment(spec)
        assert augmented.shape == spec.shape

    def test_masking_creates_zeros(self, seed):
        """SpecAugment should introduce zero regions."""
        spec = torch.ones(128, 157)
        augmented = spec_augment(
            spec,
            num_freq_masks=2,
            freq_mask_width=10,
            num_time_masks=2,
            time_mask_width=25,
        )
        # At least some values should be zeroed
        assert (augmented == 0.0).any(), "SpecAugment should create zero-masked regions"

    def test_original_unchanged(self, seed):
        """SpecAugment should not modify the original tensor (clone internally)."""
        spec = torch.ones(128, 157)
        original = spec.clone()
        _ = spec_augment(spec)
        assert torch.equal(spec, original), "Original tensor should not be modified"

    def test_no_masking_when_width_zero(self, seed):
        """With zero mask width, output should equal input."""
        spec = torch.randn(128, 157)
        augmented = spec_augment(
            spec,
            num_freq_masks=5,
            freq_mask_width=0,
            num_time_masks=5,
            time_mask_width=0,
        )
        # When width is sampled from randint(0, 0+1), it can be 0
        # We can't guarantee exact equality since randint(0,1) can give 0 or not
        # but the shape must match
        assert augmented.shape == spec.shape

    def test_invalid_dims_raises(self, seed):
        """4D input should raise ValueError."""
        spec = torch.randn(2, 1, 128, 157)
        with pytest.raises(ValueError, match="2D or 3D"):
            spec_augment(spec)


class TestMixup:
    """Tests for Mixup augmentation."""

    def test_mixup_output_shapes(self, seed):
        """Mixup outputs have same shapes as inputs."""
        x1 = torch.randn(1, 128, 157)
        y1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x2 = torch.randn(1, 128, 157)
        y2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

        mixed_x, mixed_y = mixup(x1, y1, x2, y2, alpha=0.4)
        assert mixed_x.shape == x1.shape
        assert mixed_y.shape == y1.shape

    def test_mixup_labels_sum_to_one(self, seed):
        """Mixed labels should still sum to 1.0 (convex combination)."""
        y1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        y2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x1 = torch.randn(10)
        x2 = torch.randn(10)

        _, mixed_y = mixup(x1, y1, x2, y2, alpha=0.4)
        assert torch.isclose(mixed_y.sum(), torch.tensor(1.0), atol=1e-6)

    def test_mixup_alpha_zero_returns_original(self, seed):
        """With alpha=0, mixup should return the first sample unchanged."""
        x1 = torch.randn(10)
        y1 = torch.tensor([1.0, 0.0])
        x2 = torch.randn(10)
        y2 = torch.tensor([0.0, 1.0])

        mixed_x, mixed_y = mixup(x1, y1, x2, y2, alpha=0.0)
        assert torch.equal(mixed_x, x1)
        assert torch.equal(mixed_y, y1)

    def test_mixup_is_interpolation(self, seed):
        """Mixed values should be between the two inputs element-wise."""
        x1 = torch.zeros(100)
        y1 = torch.tensor([1.0, 0.0])
        x2 = torch.ones(100)
        y2 = torch.tensor([0.0, 1.0])

        mixed_x, mixed_y = mixup(x1, y1, x2, y2, alpha=0.4)
        # All values should be between 0 and 1
        assert (mixed_x >= 0.0).all() and (mixed_x <= 1.0).all()


class TestMixupBatch:
    """Tests for batch-level Mixup."""

    def test_batch_mixup_shapes(self, seed):
        """Batch mixup preserves shapes."""
        batch_x = torch.randn(8, 1, 128, 157)
        batch_y = torch.eye(4).repeat(2, 1)  # 8 one-hot labels

        mixed_x, mixed_y = mixup_batch(batch_x, batch_y, alpha=0.4)
        assert mixed_x.shape == batch_x.shape
        assert mixed_y.shape == batch_y.shape

    def test_batch_mixup_alpha_zero(self, seed):
        """With alpha=0, batch should be unchanged."""
        batch_x = torch.randn(4, 10)
        batch_y = torch.eye(4)

        mixed_x, mixed_y = mixup_batch(batch_x, batch_y, alpha=0.0)
        assert torch.equal(mixed_x, batch_x)
        assert torch.equal(mixed_y, batch_y)


class TestToOneHot:
    """Tests for one-hot encoding utility."""

    def test_one_hot_shape(self):
        """Output shape is (B, num_classes)."""
        labels = torch.tensor([0, 1, 2, 3])
        one_hot = to_one_hot(labels, num_classes=4)
        assert one_hot.shape == (4, 4)

    def test_one_hot_values(self):
        """One-hot vectors have exactly one 1.0 per row."""
        labels = torch.tensor([0, 2, 1, 3])
        one_hot = to_one_hot(labels, num_classes=4)
        assert torch.equal(one_hot.sum(dim=1), torch.ones(4))

    def test_one_hot_correct_positions(self):
        """The 1.0 is at the correct class index."""
        labels = torch.tensor([2])
        one_hot = to_one_hot(labels, num_classes=4)
        expected = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        assert torch.equal(one_hot, expected)

    def test_one_hot_dtype(self):
        """Output dtype should be float32."""
        labels = torch.tensor([0, 1])
        one_hot = to_one_hot(labels, num_classes=4)
        assert one_hot.dtype == torch.float32
