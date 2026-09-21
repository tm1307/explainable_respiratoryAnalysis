# Conference Submission Roadmap & Teammate Task Guide

## Current State: What Is Built

The pipeline foundation is complete and covered by 90+ unit tests:

| Component | File | Status |
|-----------|------|--------|
| Log-Mel extraction | `src/features/mel_features.py` | ✅ Done |
| ICBHI data loader (patient-independent GroupKFold) | `src/data/icbhi_dataset.py` | ✅ Done |
| Noise injection (math-verified SNR) | `src/data/noise_bank.py` | ✅ Done |
| BaselineCNN + Attention Pooling + FocalLoss | `src/models/baseline_cnn.py` | ✅ Done |
| Training loop (SpecAugment, noise-aware, FocalLoss) | `src/pipeline/train.py` | ✅ Done |
| Grad-CAM + SHAP GradientExplainer | `src/explainability/` | ✅ Done |
| Faithfulness metrics (Insertion/Deletion AUC, AOPC) | `src/metrics/faithfulness.py` | ✅ Done |
| Stability metrics (SSIM, Spearman, IoU) | `src/metrics/stability.py` | ✅ Done |
| Joint Reliability Index (JRI) + AURC | `src/metrics/jri.py` | ✅ Done |
| Interactive dashboard (4-tab clinical UI) | `src/dashboard/app.py` | ✅ Done |

---

## Environment Setup (All Contributors — Do This First)

```bash
# 1. Clone and enter the repo
cd micro_project

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify everything works
python -m pytest tests/ -v
# Expected: 90 tests pass

# 5. Run a quick training sanity check (takes ~3 min, no dataset needed)
python -m src.pipeline.train --synthetic --epochs 5

# 6. Launch the dashboard
streamlit run src/dashboard/app.py
```

---

## Hard Experiments Required for Conference (IEEE EMBC / ICASSP Level)

---

### Task A — Baseline vs Noise-Aware Training Grid

**Assigned to:** `[Team Member Name]`  
**Hypothesis being tested:** H1 (Robustness) and H2 (Stability)  
**Expected output:** `models/baseline_clean.pt`, `models/baseline_noiseaware.pt`, `results/snr_sweep_table.csv`

#### Step 1 — Train the Clean Baseline

```bash
python -m src.pipeline.train \
    --icbhi \
    --epochs 50 \
    --batch-size 32 \
    --seed 42
```

This uses the ICBHI dataset in `data/icbhi/` (patient-independent fold 0), trains with **SpecAugment only** (no noise injection), saves:
- `models/baseline_cnn.pt` — best checkpoint
- `models/training_history.json` — epoch-by-epoch loss + F1
- `models/eval_results.json` — final val metrics

Rename output: `cp models/baseline_cnn.pt models/baseline_clean.pt`

#### Step 2 — Train the Noise-Aware Model

```bash
python -m src.pipeline.train \
    --icbhi \
    --noise-aware \
    --epochs 50 \
    --batch-size 32 \
    --seed 42
```

`--noise-aware` activates `NoisyBatchTransform` in `src/pipeline/train.py`, injecting random-SNR Gaussian noise (0–20 dB) on every batch during training. Rename: `cp models/baseline_cnn.pt models/baseline_noiseaware.pt`

#### Step 3 — Run the SNR Sweep Evaluation

Create `scripts/run_snr_sweep.py` (use the skeleton below):

```python
# scripts/run_snr_sweep.py
import torch, json
from src.models.baseline_cnn import BaselineCNN
from src.pipeline.inference import InferencePipeline
from src.data.noise_bank import mix_at_snr
# ... load test set, iterate snr_levels = [20,15,10,5,0,-5], collect F1 per SNR
# Save: results/snr_sweep_table.csv with columns [model, snr_db, f1_macro, shap_ssim, jri]
```

Call `src.metrics.jri.compute_full_reliability_profile()` to get JRI per SNR.

#### Deliverable

A CSV at `results/snr_sweep_table.csv` and a plot: **F1 vs SNR** and **SHAP SSIM vs SNR** (two y-axes, two model lines each). The dashboard Tab 3 already renders these if you pass the CSV path.

#### How to verify it worked

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/snr_sweep_table.csv')
print(df.groupby('model')['f1_macro'].describe())
"
```

---

### Task B — Defending JRI: Decoupling Case Study

**Assigned to:** `[Team Member Name]`  
**Goal:** Prove JRI is strictly better than naive averaging R+S/2 for detecting models that "predict correctly but explain garbage"  
**Output file:** `results/jri_ablation_table.json`

#### Step 1 — Find Decoupled Instances

After Task A's SNR sweep is done, run this search over the test set:

```python
# In scripts/find_decoupled_cases.py
# For each test clip at SNR=5dB:
#   - Compute R(SNR=5) = f1_noisy / f1_clean  (use src.metrics.jri.normalized_robustness)
#   - Compute S(SNR=5) = ssim(shap_clean, shap_noisy)  (use src.metrics.stability.ssim_2d)
#   - Flag if R > 0.8 AND S < 0.4
# Print how many clips fall into this "silent failure" zone
```

#### Step 2 — Compute Metric Comparison Table

For each flagged clip, record four scores:

| Metric | Formula | Value |
|--------|---------|-------|
| F1 alone | f1_noisy | (high — model looks fine) |
| Stability alone | SSIM(shap_clean, shap_noisy) | (low — explanation degraded) |
| Naive mean | (R + S) / 2 | (artificially inflated) |
| JRI (ours) | 2·R·S / (R + S) | (correctly penalised) |

Call `src.metrics.jri.joint_reliability_index(R, S)` — this is the key function.

#### Deliverable

A table (saved as `results/jri_ablation_table.json`) showing the above four metrics for at least 5 clips where naive averaging masks the failure and JRI exposes it. This is the ablation row in the paper.

#### How to verify it worked

JRI must be meaningfully lower than `(R+S)/2` for the flagged clips. If `JRI ≈ (R+S)/2`, the cases aren't decoupled enough — try a lower SNR (e.g., 0 dB).

---

### Task C — Cross-Corpus Generalisation (HF_Lung_V1)

**Assigned to:** `[Team Member Name]`  
**Goal:** Zero-shot eval on a second dataset to expose the generalisation gap  
**Output file:** `results/cross_corpus_results.json`

#### Step 1 — Write the HF_Lung Dataset Wrapper

Create `src/data/hf_lung_dataset.py` following the **exact same interface** as `ICBHIDataset`:

```python
class HFLungDataset(Dataset):
    """
    PyTorch Dataset for HF_Lung_V1.
    Download: https://github.com/ryersonmultimedialab/HF_Lung_V1

    Class mapping must match ICBHI:
      0 = Normal, 1 = Crackle, 2 = Wheeze, 3 = Both
    HF_Lung uses 'I' (inspiration) / 'E' (expiration) + annotation events.
    Map: no events → 0, crackle event → 1, wheeze event → 2, both → 3.
    """
    def __init__(self, audio_dir, annotation_dir, sample_rate=16000, duration_sec=5.0):
        ...
    def __len__(self): ...
    def __getitem__(self, idx): ...  # returns (mel_spec_tensor, label)
```

The `__getitem__` must return `(1, n_mels, T)` mel spectrograms so it works with the existing `InferencePipeline` without modification.

#### Step 2 — Zero-Shot Evaluation

```bash
# Load models/baseline_noiseaware.pt (from Task A, no retraining on HF_Lung)
python scripts/eval_cross_corpus.py \
    --model models/baseline_noiseaware.pt \
    --dataset hf_lung \
    --data-dir data/hf_lung/ \
    --output results/cross_corpus_results.json
```

Use `src.models.classifier.evaluate()` — it already computes all needed metrics.

#### Deliverable

`results/cross_corpus_results.json` with:
- F1 macro on ICBHI test set (from Task A)
- F1 macro on HF_Lung (zero-shot)
- JRI on both datasets at SNR=5dB

We report the **drop** — not hiding it, but quantifying it with our metrics.

---

### Task D — Real Hospital Noise Injection

**Assigned to:** `[Team Member Name]`  
**Goal:** Compare JRI under synthetic Gaussian noise vs real non-stationary hospital noise  
**Output file:** `results/real_noise_comparison.json`

#### Step 1 — Curate Noise Bank

Collect 10–15 minutes of real hospital/clinical ambient noise. Sources:
- **ESC-50** hospital subset (hospital, typing, clock, footsteps): `pip install datasets` then `datasets.load_dataset("ashraq/esc50")`
- **FreeSound** query: "hospital ambient", "ICU monitor beep", "stethoscope noise"

Save noise clips to `data/noise/hospital/` as 16 kHz mono WAV files.

#### Step 2 — Load via NoiseBank

`src/data/noise_bank.py` already has `mix_at_snr()`. Just pass a real noise waveform instead of Gaussian:

```python
from src.data.noise_bank import mix_at_snr
import torchaudio

noise_waveform, sr = torchaudio.load("data/noise/hospital/icu_monitor.wav")
noisy_signal = mix_at_snr(clean_signal, noise_waveform.squeeze(), snr_db=5.0)
```

#### Step 3 — Comparative Evaluation

Run the same SNR sweep from Task A but with real noise instead of Gaussian at SNR=5dB. Compare:

| Noise Type | JRI @ 5dB | F1 @ 5dB | SHAP SSIM @ 5dB |
|-----------|-----------|----------|-----------------|
| Gaussian (synthetic) | ? | ? | ? |
| Hospital ambient (real) | ? | ? | ? |

#### Deliverable

`results/real_noise_comparison.json` with the above table. Typically real noise degrades JRI more than synthetic Gaussian because it's non-stationary — this strengthens the clinical relevance argument.

---

## Commit & PR Workflow

```bash
# Before every commit
python -m pytest tests/ -v
# Must pass: all existing tests (do not break what's there)

# Branch naming
git checkout -b task-A/snr-sweep-[yourname]
git checkout -b task-B/jri-ablation-[yourname]
git checkout -b task-C/cross-corpus-[yourname]
git checkout -b task-D/real-noise-[yourname]

# Commit format
git commit -m "Task A: add snr_sweep.py, results/snr_sweep_table.csv"
```

## Experiment Tracking

Use W&B or MLflow when running the long sweep loops:

```python
import wandb
wandb.init(project="respiratory-xai", name="baseline_clean_snr_sweep")
wandb.log({"snr_db": snr, "f1_macro": f1, "jri": jri})
```

Do **not** rely on terminal output for multi-hour runs.

## Data Policy

- **NEVER commit ICBHI audio files** to git (patient data, licensing restrictions)
- Add `data/icbhi/` and `data/hf_lung/` to `.gitignore`
- Results CSVs and JSON summaries are fine to commit
- Model `.pt` files: commit only if < 50 MB (use Git LFS otherwise)
