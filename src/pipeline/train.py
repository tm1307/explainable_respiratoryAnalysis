"""
Training script for BaselineCNN on respiratory sound data.

Supports:
  --icbhi      : Train on real ICBHI data (requires data/icbhi/ to exist)
  --synthetic  : Train on realistic synthetic data matching ICBHI class distribution
  --epochs N   : Number of training epochs (default: 30)
  --batch-size N

Outputs saved to models/:
  baseline_cnn.pt          — best model weights (by macro F1)
  training_history.json    — per-epoch train loss + val F1 for dashboard plots
  class_weights.json       — inverse-frequency class weights used during training
  eval_results.json        — final val-set confusion matrix + per-class F1
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchaudio.transforms as T

from src.models.baseline_cnn import BaselineCNN, FocalLoss
from src.models.classifier import train_model, evaluate
from src.config import get_default_config


# ---------------------------------------------------------------------------
# ICBHI-realistic class distribution (from the published dataset statistics)
# Normal: 53.5%, Crackle: 34.7%, Wheeze: 7.5%, Both: 4.3%
# ---------------------------------------------------------------------------
ICBHI_CLASS_PROBS = [0.535, 0.347, 0.075, 0.043]


class SpecAugment(nn.Module):
    """
    SpecAugment: frequency and time masking for mel spectrograms.
    Applied during training to improve generalisation.

    Based on: Park et al., "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition" (2019).
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 30,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_param) for _ in range(n_time_masks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x shape (B, 1, n_mels, T)"""
        for mask in self.freq_masks:
            x = mask(x)
        for mask in self.time_masks:
            x = mask(x)
        return x


def make_realistic_synthetic_data(
    n_train: int = 600,
    n_val: int = 120,
    n_mels: int = 128,
    time_frames: int = 157,
    class_probs=None,
    seed: int = 42,
):
    """
    Generate synthetic mel spectrograms with realistic ICBHI class distribution.

    Each class has slightly different spectral characteristics to make the
    training signal non-trivial:
      - Normal: low-energy, relatively flat spectrum
      - Crackle: high-frequency bursts (top mel bins boosted)
      - Wheeze: narrow band energy in mid-frequency range
      - Both: combination of crackle + wheeze patterns

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_mels: Mel bin count.
        time_frames: Time frame count (must be divisible by 16 for CNN pooling).
        class_probs: Class probability distribution.
        seed: Random seed.

    Returns:
        (train_loader, val_loader, class_weights) DataLoaders and weight tensor.
    """
    if class_probs is None:
        class_probs = ICBHI_CLASS_PROBS

    torch.manual_seed(seed)
    np.random.seed(seed)

    def _make_sample(label: int, n_mels: int, T: int) -> torch.Tensor:
        """Synthesise a mel spectrogram with class-specific spectral patterns."""
        # Base: log-normal noise (realistic mel spectrogram background)
        spec = torch.randn(1, n_mels, T) * 0.5 - 2.0  # log-scale, low energy

        if label == 0:  # Normal — relatively flat, low energy
            pass

        elif label == 1:  # Crackle — random high-frequency bursts
            # Boost top 30% of mel bins at random time positions
            hi_start = int(0.7 * n_mels)
            n_bursts = np.random.randint(3, 8)
            for _ in range(n_bursts):
                t_start = np.random.randint(0, max(1, T - 8))
                t_width = np.random.randint(2, 8)
                spec[0, hi_start:, t_start : t_start + t_width] += np.random.uniform(1.5, 3.0)

        elif label == 2:  # Wheeze — sustained mid-band narrowband energy
            mid_center = np.random.randint(int(0.3 * n_mels), int(0.6 * n_mels))
            bw = np.random.randint(3, 8)
            lo, hi = max(0, mid_center - bw), min(n_mels, mid_center + bw)
            spec[0, lo:hi, :] += np.random.uniform(1.0, 2.5)

        elif label == 3:  # Both — crackle + wheeze patterns
            hi_start = int(0.7 * n_mels)
            spec[0, hi_start:, : T // 3] += np.random.uniform(1.0, 2.0)
            mid_center = np.random.randint(int(0.3 * n_mels), int(0.6 * n_mels))
            spec[0, mid_center - 3 : mid_center + 3, :] += np.random.uniform(1.0, 2.0)

        return spec

    def _generate_split(n: int) -> tuple:
        class_counts = np.random.multinomial(n, class_probs)
        xs, ys = [], []
        for label, count in enumerate(class_counts):
            for _ in range(count):
                xs.append(_make_sample(label, n_mels, time_frames))
                ys.append(label)
        xs = torch.stack(xs)
        ys = torch.tensor(ys, dtype=torch.long)
        # Shuffle
        perm = torch.randperm(len(ys))
        return xs[perm], ys[perm]

    train_x, train_y = _generate_split(n_train)
    val_x, val_y = _generate_split(n_val)

    # Compute inverse-frequency class weights from training distribution
    counts = torch.zeros(4)
    for label in train_y:
        counts[label] += 1
    counts = torch.clamp(counts, min=1.0)
    class_weights = counts.sum() / (4.0 * counts)

    return train_x, train_y, val_x, val_y, class_weights


def load_icbhi_data(config, batch_size: int):
    """Load real ICBHI dataset from data/icbhi/."""
    from src.data.icbhi_dataset import ICBHIDataset, create_patient_independent_splits
    from src.features.mel_features import MelSpectrogramExtractor
    import glob

    audio_dir = config.icbhi_dir
    ann_dir = config.icbhi_dir
    wav_files = [os.path.basename(f) for f in glob.glob(os.path.join(audio_dir, "*.wav"))]

    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in {audio_dir}. "
            "Download ICBHI 2017 from https://bhichallenge.med.auth.gr/ and extract to data/icbhi/"
        )

    splits = create_patient_independent_splits(wav_files, k_folds=5, seed=42)
    train_files, val_files = splits[0]  # Use fold 0

    extractor = MelSpectrogramExtractor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.mel.n_fft,
        hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels,
    )

    train_ds = ICBHIDataset(audio_dir, ann_dir, train_files,
                             sample_rate=config.audio.sample_rate,
                             duration_sec=config.audio.duration_sec,
                             transform=extractor.extract)
    val_ds = ICBHIDataset(audio_dir, ann_dir, val_files,
                           sample_rate=config.audio.sample_rate,
                           duration_sec=config.audio.duration_sec,
                           transform=extractor.extract)

    class_weights = train_ds.get_class_weights()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_weights


class NoisyBatchTransform:
    """On-the-fly random SNR noise injection for noise-aware training."""

    def __init__(self, snr_min: float = 0.0, snr_max: float = 20.0):
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, batch_x: torch.Tensor) -> torch.Tensor:
        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        noise = torch.randn_like(batch_x)
        signal_power = batch_x.pow(2).mean()
        noise_power = noise.pow(2).mean().clamp(min=1e-8)
        scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        return batch_x + scale * noise


def main():
    parser = argparse.ArgumentParser(
        description="Train BaselineCNN on respiratory sounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.train --synthetic --epochs 30
  python -m src.pipeline.train --icbhi --epochs 50 --batch-size 32
        """,
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use realistic synthetic data (ICBHI class distribution)")
    parser.add_argument("--icbhi", action="store_true",
                        help="Train on real ICBHI dataset (data/icbhi/ must exist)")
    parser.add_argument("--noise-aware", action="store_true",
                        help="Enable on-the-fly SNR noise injection during training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_default_config()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.icbhi:
        print("Loading ICBHI dataset...")
        train_loader, val_loader, class_weights = load_icbhi_data(config, args.batch_size)
        time_frames = None  # determined by dataset
        print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    elif args.synthetic:
        print("Generating realistic synthetic data (ICBHI class distribution)...")
        # time_frames must be divisible by 16 (4× maxpool with stride 2)
        # 5s @ 16kHz, hop=512 → floor((80000-1024)/512)+1 = 153 → round to 160
        time_frames = 160
        train_x, train_y, val_x, val_y, class_weights = make_realistic_synthetic_data(
            n_train=800,
            n_val=160,
            n_mels=config.mel.n_mels,
            time_frames=time_frames,
            seed=args.seed,
        )
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=args.batch_size,
            shuffle=False,
        )
        print(f"Train: {len(train_x)} | Val: {len(val_x)}")
        print("Class distribution (train):")
        for i, name in enumerate(["Normal", "Crackle", "Wheeze", "Both"]):
            count = (train_y == i).sum().item()
            print(f"  {name}: {count} ({100*count/len(train_y):.1f}%)")

    else:
        print("Error: specify --synthetic or --icbhi. Run with --help for usage.")
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BaselineCNN(
        num_classes=config.model.num_classes,
        n_mels=config.mel.n_mels,
        dropout=config.model.dropout,
    ).to(device)

    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=config.model.focal_loss_gamma)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Augmentation ──────────────────────────────────────────────────────────
    spec_aug = SpecAugment(
        freq_mask_param=config.augment.spec_augment_freq_width,
        time_mask_param=config.augment.spec_augment_time_width,
        n_freq_masks=config.augment.spec_augment_freq_masks,
        n_time_masks=config.augment.spec_augment_time_masks,
    ).to(device)

    noise_fn = NoisyBatchTransform(
        snr_min=config.noise.snr_train_range[0],
        snr_max=config.noise.snr_train_range[1],
    ) if args.noise_aware else None

    def augment_fn(batch_x: torch.Tensor) -> torch.Tensor:
        x = spec_aug(batch_x)
        if noise_fn is not None:
            x = noise_fn(x)
        return x

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nStarting training for up to {args.epochs} epochs...")
    print(f"  SpecAugment: ON")
    print(f"  Noise-Aware: {'ON' if args.noise_aware else 'OFF'}")
    print(f"  Loss: FocalLoss(gamma={config.model.focal_loss_gamma}, alpha=class_weights)")
    print()

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=config.train.patience,
        noise_fn=augment_fn,
        num_classes=config.model.num_classes,
    )

    # Step scheduler after training (for LR tracking only)
    for _ in range(len(history)):
        scheduler.step()

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    # Model weights
    weights_path = "models/baseline_cnn.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"\nModel saved: {weights_path}")

    # Training history (for dashboard curves)
    history_path = "models/training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")

    # Class weights (for dashboard info)
    weights_json_path = "models/class_weights.json"
    with open(weights_json_path, "w") as f:
        json.dump(
            {
                "Normal": float(class_weights[0]),
                "Crackle": float(class_weights[1]),
                "Wheeze": float(class_weights[2]),
                "Both": float(class_weights[3]),
            },
            f,
            indent=2,
        )
    print(f"Class weights saved: {weights_json_path}")

    # Final eval results (for dashboard confusion matrix)
    print("\nRunning final evaluation on validation set...")
    final_metrics = evaluate(model, val_loader, criterion, device, config.model.num_classes)
    eval_path = "models/eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Eval results saved: {eval_path}")

    # Summary
    best_epoch = max(history, key=lambda x: x.get("val_f1_macro", 0))
    print(f"\n{'='*50}")
    print(f"Best epoch: {best_epoch['epoch']}")
    print(f"  Val F1 (macro): {best_epoch.get('val_f1_macro', 0):.4f}")
    print(f"  Val Accuracy:   {best_epoch.get('val_accuracy', 0):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
