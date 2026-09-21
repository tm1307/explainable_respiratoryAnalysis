# 6. CURRENT RESULTS AND PROTOTYPE STATUS 

## 6.1 Dataset and Split Summary 

**Table 6.1: Dataset and split summary** 

| Item | Detail |
|------|--------|
| **Dataset** | Public ICBHI-style respiratory sound corpus (synthetic subset for prototype) |
| **Classes** | Normal, Wheeze, Crackle, Both/Other |
| **Total recordings** | 960 clips (simulated class-imbalanced set) |
| **Split strategy** | Patient-independent (train / val / test) |
| **Prototype subset used** | 800 Train / 160 Val used for current end-to-end run |

![Figure 6.1](screenshots/00_class_distribution.png)  
*Figure 6.1: Class distribution of the respiratory sound dataset (Normal: 54.9%, Crackle: 33.5%, Wheeze: 8.0%, Both: 3.6%)*

## 6.2 Noise Conditions Used for Robustness Testing 

**Table 6.2: Noise conditions used for robustness testing** 

| Condition | Target SNR | Purpose |
|-----------|------------|---------|
| **Clean** | — | Reference / upper-bound performance |
| **Mild** | ≈ 15 dB | Light ambient noise |
| **Moderate** | ≈ 5 dB | Typical clinical / home environment |
| **Heavy** | ≤ 0 dB | Worst-case, high-noise condition |

## 6.3 Baseline CNN + Attention — Preliminary Results 

Metrics below are from the first training and evaluation pass on the current data subset. *(Note: The prototype currently uses a highly distinct synthetic dataset to validate the pipeline end-to-end, resulting in near-perfect preliminary scores. Real-world performance on raw ICBHI data typically falls in the 55-70% macro F1 range).*

**Table 6.3: Baseline CNN + attention preliminary results** 

| Metric | Clean Subset (Preliminary) |
|--------|----------------------------|
| **Accuracy** | 1.000 |
| **Macro F1-score** | 1.000 |
| **Sensitivity (Recall)** | 1.000 (Average across all classes) |
| **Specificity** | 1.000 (Average across all classes) |

![Figure 6.2](screenshots/09_training_history.png)  
*Figure 6.2: Baseline CNN training curves (Loss, F1, and Accuracy)*

![Figure 6.3](screenshots/11_confusion_matrix.png)  
*Figure 6.3: Confusion matrix — baseline model, clean subset*

## 6.4 Explainability — Grad-CAM and SHAP 

Grad-CAM and SHAP (GradientExplainer) attribution heatmaps have been generated for the baseline model on clean audio. An early clean-versus-noisy comparison has begun to assess how much attribution shifts once noise is introduced. This forms the reference point for the explanation-stability and faithfulness metrics planned in the next phase. 

![Figure 6.4](screenshots/12_gradcam_clean_vs_noisy.png)  
*Figure 6.4: Explanation heatmap comparison (Grad-CAM) — clean vs. 0dB noisy input. The baseline model is sensitive to heavy noise, resulting in attribution shifts that our proposed JRI metric will measure.*

## 6.5 Prototype Dashboard 

An early Streamlit dashboard layout is being scaffolded to display predictions, confidence scores, explanation heatmaps, and a live SNR-sweep robustness gauge. Full integration is planned for the next phase (Section 8). 

*(The full interactive UI consists of 4 tabs running locally via `streamlit run src/dashboard/app.py`)*

![Figure 6.5](screenshots/10_results_summary_table.png)  
*Figure 6.5: Prototype screening dashboard (work in progress) — Summary table generated from the dashboard's automated evaluation routines.*
