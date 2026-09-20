# AI-Based Respiratory Sound Screening 🫁🩺
*Noise-Aware Classification & Explainability Validation*

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

This repository hosts the implementation for evaluating the robustness and explainability of AI-based respiratory sound classification models under various noise conditions. The project uniquely approaches robustness and explainability as a **joint reliability problem**, validated via the novel **Joint Reliability Index (JRI)**.

---

## 🚀 Master Implementation Plan

### 🛠 Tech Stack Recommendations

| Component | Recommended Tooling | Justification |
|-----------|--------------------|---------------|
| **Audio I/O / DSP** | `torchaudio`, `librosa`, `soundfile` | GPU-batched spectrogram transforms in torchaudio are faster than per-file librosa calls at training scale. |
| **Primary Dataset** | ICBHI 2017 Respiratory Sound Database | Standard, has patient IDs (enables patient-independent splits), comparable to existing literature. |
| **Cross-corpus Dataset** | HF_Lung_V1 | Required for external validation of the model's robustness and generalization. |
| **Noise Sources** | MUSAN, ESC-50, DEMAND | Crucial for synthetic SNR sweep; mixing with real-world noise strengthens the robustness claim. |
| **Feature Representation** | Log-Mel spectrogram (64–128 mels) + AudioSet-pretrained CNN embeddings | Pretrained general-audio models (like PANNs CNN14) outperform small task-specific models. |
| **Robust Training** | SpecAugment + Mixup + Randomized-SNR injection | Training on noise produces the "noise-aware" claim. |
| **Explainability (XAI)** | `captum` (Integrated Gradients, Grad-CAM, etc.) | Standard for PyTorch; allows comparison of method-level stability across different explainers. |
| **XAI Evaluation** | `Quantus` toolkit | Essential for systematically quantifying explanation degradation (faithfulness, stability) under noise. |
| **Experiment Tracking** | Weights & Biases / MLflow | For managing the SNR × model × XAI-method experiment grid. |

### 📅 Phased Development Timeline

1. **Phase 1 (Data & Noise)**: Data audit, build patient-independent stratified train/val/test splits, construct noise-bank with mixing utilities.
2. **Phase 2 (Baselines)**: Feature extraction (Log-Mel + PANNs embeddings) and training the baseline CNN model on clean data.
3. **Phase 3 (Noise-Aware Model)**: Fine-tune pretrained backbone with on-the-fly SNR-randomized augmentation and SpecAugment.
4. **Phase 4 (Explainability & Reliability)**: Wrap models in Grad-CAM/Score-CAM/IG. Implement stability/faithfulness metrics and the Joint Reliability Index (JRI).
5. **Phase 5 (Experiment Grid & Dashboard)**: Execute the full SNR sweep, run statistical tests, and wire the validated models into a Streamlit interactive dashboard.

---

## 🧪 Testing & Validation Protocols

### 1. Research Validation Protocol
To ensure the claims are scientifically robust and defensible:
* **Patient-Independent Cross-Validation**: k-fold (k=5) cross-validation; ensuring no patient data overlaps between train and test splits.
* **External Validation**: Run evaluations on a second public dataset (HF_Lung_V1) without retraining to benchmark cross-corpus generalization.
* **Ablation Studies**: Isolate components (pretrained backbone vs. from-scratch, SpecAugment on/off, noise-aware training on/off, XAI methods).
* **Controlled Noise Injection**: Scale noise dynamically to target SNRs:
  ```python
  SNR_dB = 10 * log10( P_signal / P_noise )
  noise_scaled = n * sqrt( P_x / (P_n * 10**(SNR_dB/10)) )
  x_noisy = x + noise_scaled
  ```
* **Joint Reliability Index (JRI)**: Evaluate harmonic mean of normalized robustness `R(SNR)` and normalized explanation stability `S(SNR)`.

### 2. Prototype / Dashboard Validation
* **Unit Testing (`pytest`)**: Verify the 7 pipeline stages independently (e.g., quality check logic, feature extractor output shape, XAI map shapes).
* **Integration Testing**: Run the full pipeline end-to-end on held-out clips to assert schema formats (label, confidence, heatmap, reliability flag). Edge cases: silence, clipped audio.
* **Latency Benchmarking**: Test inference latency and memory footprint on CPU to validate the "resource-limited setting" claim.
* **Reproducibility Checks**: Ensure fixed seeds, pinned environment dependencies, and identical reruns.

### 3. Statistical Testing
Do not report a single number without statistical significance:
* **Wilcoxon signed-rank test**: Paired comparisons for JRI/F1 across the same test clips at each SNR.
* **McNemar's test**: Paired comparison of correct/incorrect decisions.
* **Bootstrap 95% CIs**: Compute confidence intervals on all reported metrics (F1, stability, JRI).
* **Cohen's d**: Report effect size for robustness and stability gaps.

---
*This repository and methodology are built to shift respiratory AI from theoretical accuracy to robust clinical explainability.*
