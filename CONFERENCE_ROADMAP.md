# Conference Submission Roadmap & Team Contribution Guide

## 1. What Is Done (The Pipeline Foundation)
The entire core methodology has been mapped into code with strict deterministic behavior and validated by 90 passing unit tests. 
The codebase in `src/` now provides:
* **Audio & Feature processing:** `MelSpectrogramExtractor` correctly pads/truncates and extracts log-mel features.
* **Data loader (`ICBHIDataset`):** 100% implements **Patient-Independent Group K-Fold splitting**. This resolves the identity-leakage problem standard in naive implementations.
* **Noise Injection (`NoiseBank`):** Math-verified SNR scaling (`noise_scaled = n * sqrt(P_x / (P_n * 10^(SNR_dB/10)))`). Tests confirm precise decibel-level accuracy.
* **Models (`BaselineCNN`):** 4-block CNN with **Attention Pooling** and **Focal Loss** for the extreme ICBHI class imbalance. Includes Grad-CAM target layers.
* **Explainability (`GradCAM`, `IntegratedGradients`):** Implemented natively with hooks and interpolation, directly generating spatial heatmaps.
* **Metrics:** 
  * Classification (F1, AUROC, AUPRC)
  * Calibration (ECE, Brier Score)
  * Faithfulness (Insertion AUC, Deletion AUC, AOPC)
  * Stability (SSIM, Spearman, Top-K IoU)
  * **Joint Reliability Index (JRI)** and AURC calculation.
* **Dashboard (`app.py`):** End-to-end Streamlit app with interactive SNR injection.

## 2. Hard Tech Problems Remaining (How Team Mates Can Contribute)

We have the framework. To clear conference-level review (e.g., IEEE EMBC, ICASSP), the team needs to execute on the following core experiments. 

### A. The Baseline vs. Noise-Aware Training Grid
**Owner:** [Assignee]
**Problem:** We need to prove `H1` (Robustness) and `H2` (Stability).
**Tasks:**
1. **Train the Control:** Run `src/models/classifier.py` on the clean ICBHI data splits. This produces `BaselineCNN_Clean`.
2. **Train the Noise-Aware Model:** Modify the training loop to inject `NoiseBank` at random SNRs `[0, 20]dB` during training alongside `SpecAugment` and `Mixup`. This produces `BaselineCNN_Robust`.
3. **Execute the SNR Sweep:** Evaluate both models across `{20, 15, 10, 5, 0, -5} dB` SNR using the `get_snr_sweep` utility. 
4. **Deliverable:** Generate the Robustness Curve (F1 vs SNR) and Stability Curve (SSIM vs SNR).

### B. Defending the Joint Reliability Index (JRI)
**Owner:** [Assignee]
**Problem:** Reviewers will attack JRI as "just arithmetic." We need to prove it mathematically penalizes models that mask failures.
**Tasks:**
1. **Decoupling Case Study:** Using the output from the SNR Sweep, programmatically search for test instances where `Robustness (R) > 0.8` but `Stability (S) < 0.4` (i.e. model predicts correctly, but the attention map is destroyed by noise).
2. **Metric Comparison:** Show that naive averaging `(R+S)/2` keeps the score artificially high, while the JRI harmonic mean `2*R*S/(R+S)` heavily penalizes the lack of stability.
3. **Deliverable:** Ablation table row showing "F1 alone," "Stability alone," "mean(R,S)," and "JRI (ours)."

### C. Generalization Gap / Cross-Corpus Validation
**Owner:** [Assignee]
**Problem:** Reviewers highlight that ICBHI models fail on new datasets.
**Tasks:**
1. Download **HF_Lung_V1** (or a similar dataset).
2. Write a PyTorch Dataset wrapper (`hf_lung_dataset.py`) mirroring the structure of `ICBHIDataset`.
3. **Zero-shot Evaluation:** Run `BaselineCNN_Robust` on HF_Lung_V1 *without retraining*. 
4. **Deliverable:** Report the drop in JRI and F1. We are not trying to hide the drop; we are quantifying it using our metrics.

### D. The Real-Noise Injection Experiment
**Owner:** [Assignee]
**Problem:** Synthetic white/pink noise sweeps are standard, but clinical environments have specific impulse noises (talking, monitors, bumping the stethoscope).
**Tasks:**
1. Curate a mini-bank of 10-15 minutes of real ambient hospital noise (e.g., ESC-50 hospital classes, freesound).
2. Use `NoiseBank.add_noise_at_snr` to inject this specific non-stationary noise.
3. **Deliverable:** A table comparing JRI on Synthetic SNR=5dB vs Real Hospital Noise SNR=5dB.

## 3. Workflow for Contributors
1. **Environment:** 
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   pip install torchcodec # Required for torchaudio.save in tests
   ```
2. **Testing:** Run `python -m pytest tests/ -v` before committing any new code. Ensure all 90 tests pass.
3. **Tracking:** Use `Weights & Biases` (or `MLflow`) when running the SNR sweep loops. Do not rely on local terminal outputs.
4. **Data:** Download ICBHI manually, extract it to `data/icbhi/`, and ensure you never break the `GroupKFold` patient-independent split in `icbhi_dataset.py`.
