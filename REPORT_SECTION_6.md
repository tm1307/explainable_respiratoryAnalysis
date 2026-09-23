# 6. Current Results and Prototype Status

## 6.1 Dataset and Split Summary

**Table 6.1: Dataset and split summary**

| Item | Detail |
|------|--------|
| Dataset | ICBHI-distribution respiratory sound corpus |
| Classes | Normal, Crackle, Wheeze, Both |
| Total recordings | 960 clips (class-imbalanced, ICBHI proportions) |
| Split strategy | Patient-independent GroupKFold (train / val) |
| Subset used | 800 train / 160 validation |

Class distribution follows real ICBHI statistics: Normal 54.9%, Crackle 33.5%, Wheeze 8.0%, Both 3.6%.

![Figure 6.1](screenshots/00_class_distribution.png)
*Figure 6.1: Class distribution of the training dataset*

## 6.2 Noise Conditions Used for Robustness Testing

**Table 6.2: Noise conditions for robustness evaluation**

| Condition | Target SNR | Purpose |
|-----------|------------|---------|
| Clean | -- | Upper-bound reference |
| Mild | 15 dB | Light ambient noise |
| Moderate | 5 dB | Typical clinical environment |
| Heavy | 0 dB | Worst-case high-noise condition |

## 6.3 Baseline CNN + Attention -- Preliminary Results

Metrics from the first training pass on the realistic ambiguous subset. The model uses FocalLoss with inverse-frequency class weights and SpecAugment data augmentation.

**Table 6.3: Baseline CNN + Attention preliminary results**

| Metric | Value |
|--------|-------|
| Accuracy | 0.7688 |
| Macro F1 | 0.4945 |
| Weighted F1 | 0.7588 |
| Macro Precision | 0.5099 |
| Macro Recall | 0.4891 |

**Table 6.4: Per-class performance breakdown**

| Class | F1 | Precision | Recall | Specificity | Support |
|-------|-----|-----------|--------|-------------|---------|
| Normal | 0.8962 | 0.8542 | 0.9425 | 0.8082 | 87 |
| Crackle | 0.8000 | 0.8444 | 0.7600 | 0.9364 | 50 |
| Wheeze | 0.1818 | 0.2500 | 0.1429 | 0.9589 | 14 |
| Both | 0.1000 | 0.0909 | 0.1111 | 0.9338 | 9 |

The model performs well on majority classes (Normal, Crackle) but struggles with minority classes (Wheeze, Both) due to the severe class imbalance -- consistent with published ICBHI benchmarks where minority-class F1 typically ranges 0.10--0.30.

![Figure 6.2](screenshots/09_training_history.png)
*Figure 6.2: Training curves -- loss and validation F1 over 30 epochs*

![Figure 6.3](screenshots/11_confusion_matrix.png)
*Figure 6.3: Confusion matrix on the validation set (n=160)*

## 6.4 Explainability -- Grad-CAM and SHAP

Grad-CAM and SHAP GradientExplainer attribution heatmaps have been generated for the baseline model. A clean-versus-noisy comparison demonstrates how attribution shifts under noise, forming the reference point for explanation-stability metrics.

**Table 6.5: Faithfulness metrics (Grad-CAM, wheeze test sample)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Insertion AUC | 0.320 | Progressive reveal of top-attributed regions |
| Deletion AUC | 0.430 | Progressive removal of top-attributed regions |

**Table 6.6: SHAP frequency band importance (wheeze test sample)**

| Band | Mean SHAP | Interpretation |
|------|----------|----------------|
| Low (50--500 Hz) | 0.732 | Breath-cycle contribution |
| Mid (500--2k Hz) | 0.929 | Wheeze fundamental |
| High (2k--8k Hz) | 1.250 | Dominant -- wheeze harmonics |

![Figure 6.4](screenshots/12_gradcam_clean_vs_noisy.png)
*Figure 6.4: Grad-CAM comparison -- clean vs. 0 dB noisy input showing attribution shift*

![Figure 6.5](screenshots/05_shap_overlay_and_bands.png)
*Figure 6.5: SHAP attribution overlay and frequency band importance for wheeze sample*

## 6.5 Robustness and Joint Reliability Index

The SNR sweep evaluates model confidence and explanation stability across noise levels. The Joint Reliability Index (JRI) combines robustness R and explanation stability S via their harmonic mean: JRI = 2RS/(R+S).

![Figure 6.6](screenshots/06_snr_robustness_sweep.png)
*Figure 6.6: Confidence degradation and SHAP stability across SNR levels*

![Figure 6.7](screenshots/07_jri_vs_snr_and_ablation_table.png)
*Figure 6.7: JRI decomposition -- robustness vs. stability, with naive average comparison*

## 6.6 Prototype Dashboard

The Streamlit dashboard provides four analysis tabs:
1. **Analysis** -- audio upload, waveform, spectrogram, classification with confidence
2. **Explainability** -- side-by-side Grad-CAM and SHAP overlays, frequency band chart
3. **Robustness** -- interactive SNR sweep, JRI gauge, per-SNR result cards
4. **Model Performance** -- training curves, per-class F1, live faithfulness metrics

![Figure 6.8](screenshots/10_results_summary_table.png)
*Figure 6.8: Results summary table from the prototype evaluation pipeline*
