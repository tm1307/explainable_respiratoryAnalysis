"""Generate and save all key figures from wheeze test run to screenshots/."""
import sys, pathlib, os, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
import numpy as np
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.models.baseline_cnn import BaselineCNN
from src.pipeline.inference import InferencePipeline
from src.features.mel_features import MelSpectrogramExtractor, pad_or_truncate
from src.data.noise_bank import mix_at_snr
from src.metrics.faithfulness import insertion_auc, deletion_auc, aopc
from src.metrics.stability import ssim_2d
from src.metrics.jri import joint_reliability_index
from src.explainability.shap_explainer import compute_shap_heatmap_fast

os.makedirs("screenshots", exist_ok=True)

CLASS_NAMES  = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_COLORS = ["#2d7dd2", "#ef8c3b", "#3bb273", "#c94040"]
DARK   = "#0e1117"
CARD   = "#1a1f2e"
BORDER = "#2a3145"
TEXT   = "#e8eaf0"
MUTED  = "#8892a4"
ACCENT = "#3d9df8"
ACCENT2= "#4ecdc4"


def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


def save_fig(fig, name):
    path = f"screenshots/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {name}.png  ({size_kb:.0f} KB)")


# Load model & pipeline
model = BaselineCNN(num_classes=4, n_mels=128)
model.load_state_dict(torch.load("models/baseline_cnn.pt", map_location="cpu", weights_only=True))
model.eval()
pipeline  = InferencePipeline(model, device="cpu", enable_explainability=True)
extractor = MelSpectrogramExtractor(sample_rate=16000)

# Load wheeze WAV
waveform, _ = torchaudio.load("test_audio/wheeze.wav")
waveform    = waveform.squeeze()
wav_fixed   = pad_or_truncate(waveform, 16000 * 5)
mel_spec    = extractor.extract(wav_fixed).squeeze().numpy()
input_tensor = extractor.extract(wav_fixed).unsqueeze(0)

print("\nGenerating figures for wheeze.wav ...\n")

# 1 — Waveform
t = np.linspace(0, len(waveform) / 16000, len(waveform))
fig, ax = plt.subplots(figsize=(12, 2.8)); fig.patch.set_facecolor(DARK); style_ax(ax)
ax.fill_between(t, waveform.numpy(), alpha=0.6, color=ACCENT)
ax.plot(t, waveform.numpy(), lw=0.5, color=ACCENT)
ax.axhline(0, color=BORDER, lw=0.7)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.set_title("Wheeze — Raw Waveform  (5 s · 16 kHz · mono)", pad=8)
plt.tight_layout(pad=0.8)
save_fig(fig, "01_waveform")

# 2 — Log-Mel Spectrogram
fig, ax = plt.subplots(figsize=(12, 4)); fig.patch.set_facecolor(DARK); style_ax(ax)
im = ax.imshow(mel_spec, aspect="auto", origin="lower", cmap="magma")
ax.set_xlabel("Time Frame"); ax.set_ylabel("Mel Bin")
ax.set_title("Log-Mel Spectrogram  (128 mels · hop=512)", pad=8)
cb = plt.colorbar(im, ax=ax)
cb.ax.tick_params(labelsize=7, colors=MUTED)
cb.set_label("Log Energy", color=MUTED, fontsize=7)
plt.tight_layout(pad=0.8)
save_fig(fig, "02_log_mel_spectrogram")

# 3 — Prediction + Class Probabilities
result = pipeline.predict(waveform, run_quality_checks=False)
print(f"  Predicted: {result.label}  Confidence: {result.confidence:.1%}  Reliability: {result.reliability_flag}")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5)); fig.patch.set_facecolor(DARK)
for ax in axes:
    style_ax(ax)

rel_color = "#3bb273" if result.reliability_flag == "high" else "#f4c842"
for i, (lbl, val, col) in enumerate([
    ("Predicted Class", result.label,                    "#3d9df8"),
    ("Confidence",      f"{result.confidence:.1%}",      "#4ecdc4"),
    ("Reliability",     result.reliability_flag.upper(), rel_color),
]):
    axes[0].text(0.5, 0.78 - i*0.32, val, ha="center", va="center",
                 fontsize=22, fontweight="bold", color=col, transform=axes[0].transAxes)
    axes[0].text(0.5, 0.68 - i*0.32, lbl, ha="center", va="center",
                 fontsize=9, color=MUTED, transform=axes[0].transAxes)
axes[0].axis("off"); axes[0].set_title("Classification Result", pad=8)

probs      = list(result.class_probabilities.values())
bar_colors = [CLASS_COLORS[i] if CLASS_NAMES[i] == result.label else BORDER for i in range(4)]
bars = axes[1].barh(CLASS_NAMES, probs, color=bar_colors)
axes[1].set_xlim(0, 1.15)
axes[1].set_xlabel("Probability")
axes[1].set_title("Class Probabilities", pad=8)
for bar, val in zip(bars, probs):
    axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10, color=TEXT)
plt.tight_layout(pad=1.0)
save_fig(fig, "03_prediction_and_probabilities")

# 4 — Grad-CAM Overlay
if result.heatmap is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4)); fig.patch.set_facecolor(DARK)
    for ax in axes:
        style_ax(ax)
    axes[0].imshow(mel_spec, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Input Spectrogram", pad=8)
    axes[0].set_xlabel("Time Frame"); axes[0].set_ylabel("Mel Bin")
    axes[1].imshow(mel_spec, aspect="auto", origin="lower", cmap="magma", alpha=0.45)
    im2 = axes[1].imshow(result.heatmap, aspect="auto", origin="lower", cmap="inferno", alpha=0.6)
    axes[1].set_title(f"Grad-CAM Attribution  →  {result.label}", pad=8)
    axes[1].set_xlabel("Time Frame"); axes[1].set_ylabel("Mel Bin")
    cb2 = plt.colorbar(im2, ax=axes[1])
    cb2.ax.tick_params(labelsize=7, colors=MUTED)
    cb2.set_label("Attribution", color=MUTED, fontsize=7)
    plt.tight_layout(pad=1.0)
    save_fig(fig, "04_gradcam_overlay")

# 5 — SHAP Overlay + Band Importance
print("  Computing SHAP (CPU)...")
shap_map, band_importance = compute_shap_heatmap_fast(
    model, input_tensor, result.label_index, n_background=8
)

if shap_map is not None:
    fig = plt.figure(figsize=(16, 4.5)); fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1.3], wspace=0.35)

    ax0 = fig.add_subplot(gs[0]); style_ax(ax0)
    ax0.imshow(mel_spec, aspect="auto", origin="lower", cmap="magma")
    ax0.set_title("Input Spectrogram", pad=8)
    ax0.set_xlabel("Time Frame"); ax0.set_ylabel("Mel Bin")

    ax1 = fig.add_subplot(gs[1]); style_ax(ax1)
    ax1.imshow(mel_spec, aspect="auto", origin="lower", cmap="magma", alpha=0.45)
    im_s = ax1.imshow(shap_map, aspect="auto", origin="lower", cmap="plasma", alpha=0.65)
    ax1.set_title(f"SHAP Attribution (GradientExplainer)  →  {result.label}", pad=8)
    ax1.set_xlabel("Time Frame"); ax1.set_ylabel("Mel Bin")
    cb_s = plt.colorbar(im_s, ax=ax1)
    cb_s.ax.tick_params(labelsize=7, colors=MUTED)
    cb_s.set_label("|SHAP|", color=MUTED, fontsize=7)

    ax2 = fig.add_subplot(gs[2]); style_ax(ax2)
    short_names = ["Low\n50–500 Hz", "Mid\n500–2k Hz", "High\n2k–8k Hz"]
    vals_b  = list(band_importance.values())
    bar_col = [ACCENT, ACCENT2, "#a29bfe"]
    bars2   = ax2.bar(short_names, vals_b, color=bar_col, width=0.55)
    for bar, v in zip(bars2, vals_b):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:.3f}", ha="center", fontsize=9, color=TEXT, fontweight="bold")
    ax2.set_title("Mean |SHAP|\nper Freq Band", pad=8)
    ax2.set_ylabel("Normalised Importance")
    ax2.set_ylim(0, max(vals_b) * 1.3)

    save_fig(fig, "05_shap_overlay_and_bands")
    print(f"  Bands: Low={vals_b[0]:.3f}  Mid={vals_b[1]:.3f}  High={vals_b[2]:.3f}")

# 6 — SNR Robustness Sweep
print("  Running SNR sweep (6 levels)...")
snr_levels   = [20, 15, 10, 5, 0, -5]
confidences, labels_sw, shap_ssims = [], [], []

for snr in snr_levels:
    noise = torch.randn_like(waveform)
    noisy = mix_at_snr(waveform, noise, float(snr))
    r_n   = pipeline.predict(noisy, run_quality_checks=False)
    confidences.append(r_n.confidence)
    labels_sw.append(r_n.label)
    if shap_map is not None:
        wn = pad_or_truncate(noisy, 16000 * 5)
        sn, _ = compute_shap_heatmap_fast(model, extractor.extract(wn).unsqueeze(0),
                                           result.label_index, n_background=5)
        wc = pad_or_truncate(waveform, 16000 * 5)
        sc, _ = compute_shap_heatmap_fast(model, extractor.extract(wc).unsqueeze(0),
                                           result.label_index, n_background=5)
        shap_ssims.append(ssim_2d(sc, sn) if (sn is not None and sc is not None) else None)
    else:
        shap_ssims.append(None)

changed = [l != result.label for l in labels_sw]

fig, axes = plt.subplots(1, 2, figsize=(14, 4)); fig.patch.set_facecolor(DARK)
for ax in axes:
    style_ax(ax)

dot_c = [CLASS_COLORS[3] if c else ACCENT for c in changed]
axes[0].plot(snr_levels, confidences, color=ACCENT, lw=2.5, zorder=2)
axes[0].scatter(snr_levels, confidences, c=dot_c, s=80, zorder=3)
axes[0].axhline(0.8, color="#3bb273", ls="--", lw=1.2, label="High reliability (0.8)")
axes[0].axhline(0.5, color="#f4c842", ls=":",  lw=1.2, label="Medium reliability (0.5)")
for i, (snr, conf, lbl) in enumerate(zip(snr_levels, confidences, labels_sw)):
    axes[0].annotate(lbl, (snr, conf), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7,
                     color=CLASS_COLORS[3] if changed[i] else TEXT)
axes[0].invert_xaxis(); axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("Model Confidence")
axes[0].set_title("Confidence vs SNR  (Wheeze)", pad=8)
axes[0].legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER)

valid = [(s, v) for s, v in zip(snr_levels, shap_ssims) if v is not None]
if valid:
    xs, ys = zip(*valid)
    axes[1].plot(list(xs), list(ys), color=ACCENT2, lw=2.5, marker="o", ms=8)
    axes[1].axhline(0.7, color="#3bb273", ls="--", lw=1.2, label="Good stability (0.7)")
    axes[1].invert_xaxis(); axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("SHAP SSIM (Stability)")
    axes[1].set_title("SHAP Explanation Stability vs SNR", pad=8)
    axes[1].legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER)

plt.tight_layout(pad=1.2)
save_fig(fig, "06_snr_robustness_sweep")

# 7 — JRI Table + Curve
clean_conf = result.confidence
robust_vals, stab_vals, jri_vals, naive_vals = [], [], [], []
for conf, ssim_v in zip(confidences, shap_ssims):
    R = max(0.0, min(1.0, conf / max(clean_conf, 1e-6)))
    S = ssim_v if ssim_v is not None else 0.5
    J = joint_reliability_index(R, S)
    N = (R + S) / 2
    robust_vals.append(R); stab_vals.append(S); jri_vals.append(J); naive_vals.append(N)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5)); fig.patch.set_facecolor(DARK)
for ax in axes:
    style_ax(ax)

axes[0].plot(snr_levels, jri_vals,    color="#a29bfe", lw=2.5, marker="D", ms=8, label="JRI  2RS/(R+S)")
axes[0].plot(snr_levels, naive_vals,  color="#fdcb6e", lw=1.8, ls="--", marker="s", ms=6, label="Naive  (R+S)/2")
axes[0].plot(snr_levels, robust_vals, color=ACCENT,    lw=1.5, ls=":",  marker="o", ms=5, label="Robustness R")
axes[0].plot(snr_levels, stab_vals,   color=ACCENT2,   lw=1.5, ls="-.", marker="^", ms=5, label="Stability S")
axes[0].invert_xaxis(); axes[0].set_ylim(-0.05, 1.05)
axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("Score")
axes[0].set_title("JRI vs SNR  —  Robustness × Stability", pad=8)
axes[0].legend(fontsize=8.5, facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER)

# Table
row_labels = [f"{s} dB" for s in snr_levels]
table_data = []
for R, S, N, J in zip(robust_vals, stab_vals, naive_vals, jri_vals):
    row = [f"{R:.3f}", f"{S:.3f}", f"{N:.3f}", f"{J:.3f}"]
    table_data.append(row)
col_labels = ["R", "S", "(R+S)/2  naive", "JRI  2RS/(R+S)"]
axes[1].axis("off")
tbl = axes[1].table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                    cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1e2540" if r == 0 else (CARD if r % 2 == 0 else "#141929"))
    cell.set_edgecolor(BORDER)
    cell.set_text_props(color=ACCENT if r == 0 else TEXT)
    cell.set_height(0.1)
axes[1].set_title("JRI Ablation — Why Naive Average Misleads", pad=8)

plt.tight_layout(pad=1.2)
save_fig(fig, "07_jri_vs_snr_and_ablation_table")

# 8 — Faithfulness Metrics
print("  Computing faithfulness metrics...")
ins = dlt = aop = None
if result.heatmap is not None:
    ins = insertion_auc(model, input_tensor, result.heatmap, result.label_index, n_steps=30)
    dlt = deletion_auc(model, input_tensor, result.heatmap, result.label_index, n_steps=30)
    aop = aopc(model, input_tensor, result.heatmap, result.label_index, K=15)
    print(f"  Insertion AUC: {ins:.4f}  |  Deletion AUC: {dlt:.4f}  |  AOPC: {aop:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4)); fig.patch.set_facecolor(DARK)
    for ax in axes:
        style_ax(ax)

    names_f = ["Insertion AUC ↑", "Deletion AUC ↓", "AOPC ↑"]
    vals_f  = [ins, dlt, aop]
    bar_c   = ["#3bb273", "#c94040", ACCENT2]
    b       = axes[0].bar(names_f, vals_f, color=bar_c, width=0.5)
    for bar, v in zip(b, vals_f):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{v:.4f}", ha="center", fontsize=11, color=TEXT, fontweight="bold")
    axes[0].set_title("Grad-CAM Faithfulness Metrics  (Wheeze)", pad=8)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, max(vals_f) * 1.35)

    axes[1].axis("off")
    interp = [
        ("Insertion AUC", f"{ins:.4f}",
         "Reveal top-attributed regions progressively from blank.\nHigher → explanation captures truly important pixels."),
        ("Deletion AUC",  f"{dlt:.4f}",
         "Remove top-attributed regions progressively.\nLower → removing them drops confidence (faithful)."),
        ("AOPC",          f"{aop:.4f}",
         "Average confidence drop after removing each top-k feature.\nHigher → removing key regions hurts the model more."),
    ]
    for i, (name, val, desc) in enumerate(interp):
        y = 0.88 - i * 0.33
        axes[1].text(0.0, y,        f"{name}: {val}", color=ACCENT, fontsize=11,
                     fontweight="bold", transform=axes[1].transAxes)
        axes[1].text(0.0, y - 0.09, desc, color=MUTED, fontsize=8.5,
                     transform=axes[1].transAxes, linespacing=1.5)
    axes[1].set_title("Metric Definitions", pad=8)

    plt.tight_layout(pad=1.2)
    save_fig(fig, "08_faithfulness_metrics")

# 9 — Training History
with open("models/training_history.json") as f:
    history = json.load(f)

epochs     = [h["epoch"]        for h in history]
train_loss = [h["train_loss"]   for h in history]
val_f1     = [h["val_f1_macro"] for h in history]
val_acc    = [h["val_accuracy"] for h in history]

fig, ax1 = plt.subplots(figsize=(11, 4)); fig.patch.set_facecolor(DARK); style_ax(ax1)
ax2 = ax1.twinx()
ax2.set_facecolor(CARD)
ax2.tick_params(colors=MUTED, labelsize=9)
for sp in ax2.spines.values():
    sp.set_edgecolor("none")

ax1.plot(epochs, train_loss, color=ACCENT,    lw=2.5, label="Train Loss")
ax2.plot(epochs, val_f1,     color=ACCENT2,   lw=2.5, ls="--", label="Val F1")
ax2.plot(epochs, val_acc,    color="#a29bfe", lw=1.8, ls=":",  label="Val Accuracy")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color=ACCENT)
ax2.set_ylabel("Score", color=ACCENT2)
ax1.set_title("Training Curves — BaselineCNN (Synthetic ICBHI Distribution + SpecAugment)", pad=8)
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9,
           facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER)
plt.tight_layout(pad=1.0)
save_fig(fig, "09_training_history")

# 10 — Full Results Summary Table
with open("models/eval_results.json") as f:
    ev = json.load(f)

fig, ax = plt.subplots(figsize=(12, 5)); fig.patch.set_facecolor(DARK); ax.axis("off")

summary_rows = [
    ["Predicted Class",   result.label,                          "wheeze.wav inference"],
    ["Confidence",        f"{result.confidence:.1%}",            "Softmax max probability"],
    ["Reliability",       result.reliability_flag.upper(),       "High ≥0.8 · Mid ≥0.5 · Low <0.5"],
    ["Val F1",            f"{ev['f1_macro']:.4f}",               "4-class macro-averaged F1"],
    ["Val Accuracy",      f"{ev['accuracy']:.4f}",               "Overall accuracy on val set"],
    ["Balanced Acc.",     f"{ev['balanced_accuracy']:.4f}",      "Per-class balanced accuracy"],
    ["F1 — Normal",       f"{ev['f1_class_0']:.4f}",             "Class 0"],
    ["F1 — Crackle",      f"{ev['f1_class_1']:.4f}",             "Class 1"],
    ["F1 — Wheeze",       f"{ev['f1_class_2']:.4f}",             "Class 2"],
    ["F1 — Both",         f"{ev['f1_class_3']:.4f}",             "Class 3"],
    ["Insertion AUC",     f"{ins:.4f}" if ins else "—",          "XAI faithfulness ↑"],
    ["Deletion AUC",      f"{dlt:.4f}" if dlt else "—",          "XAI faithfulness ↓"],
    ["AOPC",              f"{aop:.4f}" if aop else "—",          "XAI faithfulness ↑"],
    ["JRI @ 5 dB",        f"{jri_vals[3]:.4f}",                  "Joint Reliability Index"],
    ["SHAP High Band",    f"{list(band_importance.values())[2]:.3f}" if band_importance else "—",
                                                                  "2k–8k Hz attribution (dominant for wheeze)"],
]
col_labels = ["Metric", "Value", "Notes"]
tbl = ax.table(
    cellText=summary_rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=[0.28, 0.2, 0.52],
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1e2540" if r == 0 else (CARD if r % 2 == 0 else "#141929"))
    cell.set_edgecolor(BORDER)
    cell.set_text_props(color=ACCENT if r == 0 else TEXT)
    cell.set_height(0.055)
ax.set_title("Complete Results Summary — Wheeze Test (wheeze.wav)",
             color=TEXT, fontsize=12, pad=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "10_results_summary_table")

# Done
print(f"\nAll figures saved to screenshots/  ({len(os.listdir('screenshots'))} files)")
