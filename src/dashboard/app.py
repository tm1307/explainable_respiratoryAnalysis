"""
Respiratory Sound Analysis Dashboard
Clinical Research Interface — Noise-Aware Classification & Explainability

Tabs:
  1. Audio Analysis     — upload, waveform, spectrogram, prediction
  2. Explainability     — Grad-CAM + SHAP heatmaps, frequency band importance
  3. Robustness         — SNR sweep, JRI gauge, confidence degradation curve
  4. Model Performance  — training curves, confusion matrix, faithfulness metrics
"""

# Path guard
# Ensures the project root is on sys.path regardless of how Streamlit is
# invoked (e.g. `streamlit run src/dashboard/app.py` from any directory).
import sys, pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# 

import io
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import torchaudio

from src.models.baseline_cnn import BaselineCNN
from src.pipeline.inference import InferencePipeline
from src.features.mel_features import MelSpectrogramExtractor, pad_or_truncate
from src.data.noise_bank import mix_at_snr
from src.metrics.faithfulness import insertion_auc, deletion_auc, aopc
from src.metrics.stability import ssim_2d
from src.metrics.jri import joint_reliability_index

try:
    from src.explainability.shap_explainer import compute_shap_heatmap_fast
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Constants
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_COLORS = {
    "Normal":  "#2d7dd2",
    "Crackle": "#ef8c3b",
    "Wheeze":  "#3bb273",
    "Both":    "#c94040",
}
PALETTE = {
    "bg":        "#0e1117",
    "card":      "#1a1f2e",
    "border":    "#2a3145",
    "accent":    "#3d9df8",
    "accent2":   "#4ecdc4",
    "text":      "#e8eaf0",
    "text_muted":"#8892a4",
    "success":   "#3bb273",
    "warning":   "#f4c842",
    "danger":    "#c94040",
}


# Page config & CSS
def apply_custom_css():
    st.markdown(
        f"""
        <style>
        /* ── Global ── */
        html, body, [data-testid="stApp"] {{
            background-color: {PALETTE['bg']};
            color: {PALETTE['text']};
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: #10141e;
            border-right: 1px solid {PALETTE['border']};
        }}
        [data-testid="stSidebar"] .stMarkdown p {{
            color: {PALETTE['text_muted']};
            font-size: 0.82rem;
        }}

        /* ── Cards ── */
        .metric-card {{
            background: {PALETTE['card']};
            border: 1px solid {PALETTE['border']};
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.7rem;
        }}
        .metric-label {{
            color: {PALETTE['text_muted']};
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }}
        .metric-value {{
            color: {PALETTE['text']};
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.1;
        }}
        .metric-sub {{
            color: {PALETTE['text_muted']};
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }}

        /* ── Section headers ── */
        .section-header {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {PALETTE['accent']};
            font-weight: 600;
            margin: 1.4rem 0 0.6rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid {PALETTE['border']};
        }}

        /* ── Reliability badge ── */
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
        }}
        .badge-high    {{ background: #1a3d2a; color: #3bb273; border: 1px solid #3bb273; }}
        .badge-medium  {{ background: #3d3112; color: #f4c842; border: 1px solid #f4c842; }}
        .badge-low     {{ background: #3d1212; color: #c94040; border: 1px solid #c94040; }}

        /* ── Tab overrides ── */
        [data-testid="stTabs"] [data-baseweb="tab"] {{
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            color: {PALETTE['text_muted']};
        }}
        [data-testid="stTabs"] [aria-selected="true"] {{
            color: {PALETTE['accent']} !important;
            border-bottom-color: {PALETTE['accent']} !important;
        }}

        /* ── Dividers ── */
        hr {{ border-color: {PALETTE['border']} !important; }}

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {{
            border: 1px dashed {PALETTE['border']};
            border-radius: 8px;
            padding: 0.5rem;
        }}

        /* ── Disclaimer ── */
        .disclaimer {{
            font-size: 0.72rem;
            color: {PALETTE['text_muted']};
            border-top: 1px solid {PALETTE['border']};
            padding-top: 0.8rem;
            margin-top: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Model loading
@st.cache_resource
def load_model_and_pipeline():
    model = BaselineCNN(num_classes=4, n_mels=128)
    weights_path = "models/baseline_cnn.pt"
    is_trained = False
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            is_trained = True
        except Exception:
            pass
    model.eval()
    pipeline = InferencePipeline(model, device="cpu", enable_explainability=True)
    return model, pipeline, is_trained


@st.cache_data
def load_training_history():
    path = "models/training_history.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_eval_results():
    path = "models/eval_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# Plot helpers
def fig_to_st(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    buf.seek(0)
    plt.close(fig)
    return buf


def set_mpl_dark(fig, axes=None):
    fig.patch.set_facecolor(PALETTE["bg"])
    ax_list = [axes] if axes is not None else fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(PALETTE["card"])
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["text_muted"])
        ax.yaxis.label.set_color(PALETTE["text_muted"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])


def plot_waveform(waveform: torch.Tensor, sample_rate: int) -> io.BytesIO:
    t = np.linspace(0, len(waveform) / sample_rate, len(waveform))
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.fill_between(t, waveform.numpy(), alpha=0.7, color=PALETTE["accent"])
    ax.plot(t, waveform.numpy(), linewidth=0.4, color=PALETTE["accent"])
    ax.axhline(0, color=PALETTE["border"], linewidth=0.6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title("Waveform", fontsize=9, pad=6)
    set_mpl_dark(fig, ax)
    return fig_to_st(fig)


def plot_spectrogram(mel_spec: np.ndarray, title="Log-Mel Spectrogram") -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(mel_spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Time Frame", fontsize=8)
    ax.set_ylabel("Mel Bin", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    cb = plt.colorbar(im, ax=ax, label="Log Energy")
    cb.ax.tick_params(labelsize=7, colors=PALETTE["text_muted"])
    cb.set_label("Log Energy", color=PALETTE["text_muted"], fontsize=7)
    set_mpl_dark(fig, ax)
    return fig_to_st(fig)


def plot_heatmap_overlay(spec: np.ndarray, heatmap: np.ndarray, title="Heatmap Overlay") -> io.BytesIO:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    axes[0].imshow(spec, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Input Spectrogram", fontsize=9)
    axes[0].set_xlabel("Time Frame", fontsize=8)
    axes[0].set_ylabel("Mel Bin", fontsize=8)

    axes[1].imshow(spec, aspect="auto", origin="lower", cmap="magma", alpha=0.45)
    axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="inferno", alpha=0.6)
    axes[1].set_title(title, fontsize=9)
    axes[1].set_xlabel("Time Frame", fontsize=8)
    axes[1].set_ylabel("Mel Bin", fontsize=8)
    set_mpl_dark(fig)
    plt.tight_layout(pad=1.0)
    return fig_to_st(fig)


def plotly_prob_bar(class_probs: dict, predicted_label: str) -> go.Figure:
    names = list(class_probs.keys())
    vals  = list(class_probs.values())
    colors = [
        CLASS_COLORS.get(n, PALETTE["accent"]) if n == predicted_label
        else "#2a3145"
        for n in names
    ]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=PALETTE["text"]),
    ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        xaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False,
                   tickfont=dict(color=PALETTE["text_muted"], size=9)),
        yaxis=dict(showgrid=False, tickfont=dict(color=PALETTE["text"], size=11)),
        font=dict(color=PALETTE["text"]),
    )
    return fig


def plotly_shap_bars(band_importance: dict) -> go.Figure:
    bands = list(band_importance.keys())
    vals  = list(band_importance.values())
    colors = [PALETTE["accent"], PALETTE["accent2"], "#a29bfe"]
    fig = go.Figure(go.Bar(
        x=bands, y=vals,
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=PALETTE["text"]),
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Mean |SHAP| per Frequency Band",
                   font=dict(size=11, color=PALETTE["text"]), x=0.5),
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        xaxis=dict(showgrid=False, tickfont=dict(color=PALETTE["text_muted"], size=10)),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_muted"], size=9)),
        font=dict(color=PALETTE["text"]),
    )
    return fig


def plotly_jri_gauge(jri: float) -> go.Figure:
    color = (PALETTE["success"] if jri >= 0.7
             else PALETTE["warning"] if jri >= 0.4
             else PALETTE["danger"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(jri * 100, 1),
        number=dict(suffix="%", font=dict(size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickfont=dict(color=PALETTE["text_muted"], size=9)),
            bar=dict(color=color, thickness=0.3),
            bgcolor=PALETTE["card"],
            bordercolor=PALETTE["border"],
            steps=[
                dict(range=[0, 40],  color="#3d1212"),
                dict(range=[40, 70], color="#3d3112"),
                dict(range=[70, 100], color="#1a3d2a"),
            ],
        ),
        title=dict(text="Joint Reliability Index (JRI)",
                   font=dict(size=12, color=PALETTE["text_muted"])),
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor=PALETTE["card"], font=dict(color=PALETTE["text"]),
    )
    return fig


def plotly_snr_curve(snr_levels: list, confidences: list, labels: list) -> go.Figure:
    fig = go.Figure()
    changed = [l != labels[0] for l in labels]
    fig.add_trace(go.Scatter(
        x=snr_levels, y=confidences, mode="lines+markers",
        line=dict(color=PALETTE["accent"], width=2.5),
        marker=dict(
            size=9,
            color=[PALETTE["danger"] if c else PALETTE["accent"] for c in changed],
            symbol=["x" if c else "circle" for c in changed],
        ),
        name="Confidence",
        hovertemplate="SNR: %{x} dB<br>Confidence: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.8, line_dash="dash", line_color=PALETTE["success"],
                  annotation_text="High reliability", annotation_font_size=9)
    fig.add_hline(y=0.5, line_dash="dot", line_color=PALETTE["warning"],
                  annotation_text="Medium reliability", annotation_font_size=9)
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Confidence vs SNR",
                   font=dict(size=12, color=PALETTE["text"]), x=0.5),
        xaxis=dict(title="SNR (dB)", autorange="reversed",
                   showgrid=True, gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_muted"])),
        yaxis=dict(title="Confidence", range=[0, 1.05],
                   showgrid=True, gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_muted"]),
                   tickformat=".0%"),
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"]),
        showlegend=False,
    )
    return fig


def plotly_training_curves(history: list) -> go.Figure:
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", None) for h in history]
    val_f1 = [h.get("val_f1_macro", None) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, name="Train Loss",
        line=dict(color=PALETTE["accent"], width=2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_f1, name="Val F1 (macro)",
        line=dict(color=PALETTE["accent2"], width=2, dash="dash"),
        yaxis="y2",
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=60, t=30, b=10),
        title=dict(text="Training Curves", font=dict(size=12, color=PALETTE["text"]), x=0.5),
        xaxis=dict(title="Epoch", showgrid=True, gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_muted"])),
        yaxis=dict(title="Loss", showgrid=True, gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_muted"]),
                   side="left"),
        yaxis2=dict(title="F1", overlaying="y", side="right",
                    range=[0, 1], tickformat=".2f",
                    tickfont=dict(color=PALETTE["text_muted"])),
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor=PALETTE["card"], bordercolor=PALETTE["border"],
                    font=dict(size=10)),
    )
    return fig


def plotly_per_class_f1(eval_results: dict) -> go.Figure:
    f1s = [eval_results.get(f"f1_class_{i}", 0.0) for i in range(4)]
    fig = go.Figure(go.Bar(
        x=CLASS_NAMES, y=f1s,
        marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
        text=[f"{v:.3f}" for v in f1s],
        textposition="outside",
        textfont=dict(size=11, color=PALETTE["text"]),
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Per-Class F1 Score", font=dict(size=11, color=PALETTE["text"]), x=0.5),
        xaxis=dict(showgrid=False, tickfont=dict(color=PALETTE["text"])),
        yaxis=dict(range=[0, 1.15], showgrid=True, gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_muted"])),
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"]),
    )
    return fig


def plotly_faithfulness_bars(ins_auc: float, del_auc: float, aopc_val: float) -> go.Figure:
    names = ["Insertion AUC ↑", "Deletion AUC ↓", "AOPC ↑"]
    vals  = [ins_auc, del_auc, aopc_val]
    colors = [PALETTE["success"], PALETTE["danger"], PALETTE["accent2"]]
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=PALETTE["text"]),
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Faithfulness Metrics (Grad-CAM)", font=dict(size=11, color=PALETTE["text"]), x=0.5),
        xaxis=dict(showgrid=False, tickfont=dict(color=PALETTE["text"])),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_muted"])),
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"]),
    )
    return fig


# Sidebar
def render_sidebar(is_trained: bool):
    st.sidebar.markdown(
        """
        <div style="padding:1rem 0 0.5rem 0;">
            <div style="font-size:1.2rem;font-weight:700;color:#e8eaf0;">
                RespiScan
            </div>
            <div style="font-size:0.7rem;color:#8892a4;margin-top:0.15rem;">
                Explainable Respiratory Sound Analysis
            </div>
        </div>
        <hr style="border-color:#2a3145;margin:0.5rem 0;">
        """,
        unsafe_allow_html=True,
    )

    model_status = "Trained model loaded" if is_trained else "Untrained (demo mode)"
    status_color = PALETTE["success"] if is_trained else PALETTE["warning"]
    st.sidebar.markdown(
        f"""
        <div style="font-size:0.72rem;color:{PALETTE['text_muted']};margin-top:0.8rem;">
            MODEL STATUS
        </div>
        <div style="font-size:0.85rem;color:{status_color};margin-bottom:1rem;">
            ● {model_status}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_trained:
        st.sidebar.info(
            "Train the model first:\n\n"
            "```bash\npython -m src.pipeline.train \\\n"
            "  --synthetic --epochs 30\n```"
        )

    st.sidebar.markdown(
        f"""
        <div style="font-size:0.72rem;color:{PALETTE['text_muted']};margin-top:1rem;">
            PIPELINE
        </div>
        <div style="font-size:0.8rem;color:{PALETTE['text']};line-height:1.8;margin-bottom:1rem;">
            1 · Quality check<br>
            2 · Log-Mel extraction<br>
            3 · BaselineCNN inference<br>
            4 · Grad-CAM + SHAP<br>
            5 · JRI reliability score
        </div>
        <hr style="border-color:#2a3145;">
        <div style="font-size:0.72rem;color:{PALETTE['text_muted']};margin-top:0.8rem;">
            DATASET · ICBHI 2017<br>
            CLASSES · Normal / Crackle / Wheeze / Both<br>
            MODEL · 4-block CNN + Attention Pool<br>
            XAI · Grad-CAM + SHAP GradientExplainer
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="disclaimer">Research prototype — not for clinical use.</div>',
        unsafe_allow_html=True,
    )


# Tab 1: Audio Analysis
def render_tab_analysis(model, pipeline):
    st.markdown('<div class="section-header">Upload Audio</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a .wav respiratory recording here",
        type=["wav"],
        help="Mono or stereo WAV, any sample rate. Will be resampled to 16 kHz.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            f"""
            <div style="text-align:center;padding:2.5rem 1rem;color:{PALETTE['text_muted']};
                        border:1px dashed {PALETTE['border']};border-radius:8px;margin-top:0.5rem;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">▸</div>
                <div style="font-size:0.9rem;">Upload a WAV file to begin analysis</div>
                <div style="font-size:0.75rem;margin-top:0.4rem;">
                    Any respiratory recording — stethoscope, chest mic, etc.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None, None, None, None

    audio_bytes = uploaded_file.read()
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform  = resampler(waveform)
        sample_rate = 16000
    waveform = waveform.squeeze()
    st.audio(audio_bytes, format="audio/wav")
    dur = len(waveform) / sample_rate
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Duration</div>'
            f'<div class="metric-value">{dur:.2f}s</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Sample Rate</div>'
            f'<div class="metric-value">16 kHz</div></div>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Samples</div>'
            f'<div class="metric-value">{len(waveform):,}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-header">Waveform</div>', unsafe_allow_html=True)
    st.image(plot_waveform(waveform, sample_rate), use_container_width=True)
    extractor = MelSpectrogramExtractor(sample_rate=16000)
    wav_fixed = pad_or_truncate(waveform, 16000 * 5)
    mel_spec  = extractor.extract(wav_fixed).squeeze().numpy()

    st.markdown('<div class="section-header">Log-Mel Spectrogram</div>', unsafe_allow_html=True)
    st.image(plot_spectrogram(mel_spec), use_container_width=True)
    st.markdown('<div class="section-header">Classification Result</div>', unsafe_allow_html=True)
    result = pipeline.predict(waveform, run_quality_checks=True)

    if result.label == "REJECTED":
        st.error(f"Audio rejected: {', '.join(result.quality_report.reasons)}")
        return None, None, None, None

    badge_cls = f"badge-{result.reliability_flag}" if result.reliability_flag in ("high","medium","low") else ""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Predicted Class</div>'
            f'<div class="metric-value" style="color:{CLASS_COLORS.get(result.label, PALETTE["text"])};">'
            f'{result.label}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Confidence</div>'
            f'<div class="metric-value">{result.confidence:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Reliability</div>'
            f'<div class="metric-value">'
            f'<span class="badge {badge_cls}">{result.reliability_flag.upper()}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plotly_prob_bar(result.class_probabilities, result.label),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    if result.quality_report:
        with st.expander("Audio Quality Report", expanded=False):
            qr = result.quality_report
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration", f"{qr.duration_sec:.2f}s")
            c2.metric("Peak Amp", f"{qr.peak_amplitude:.4f}")
            c3.metric("RMS Energy", f"{qr.rms_energy:.4f}")
            c4.metric("Silence Frac", f"{qr.silence_fraction:.2%}")

    return waveform, mel_spec, result, extractor


# Tab 2: Explainability
def render_tab_explainability(model, waveform, mel_spec, result):
    if result is None:
        st.info("Upload audio in the Analysis tab first.")
        return

    wav_fixed = pad_or_truncate(waveform, 16000 * 5)
    extractor = MelSpectrogramExtractor(sample_rate=16000)
    mel_tensor = extractor.extract(wav_fixed)
    input_tensor = mel_tensor.unsqueeze(0)  # (1, 1, 128, T)

    col_left, col_right = st.columns(2)

    # Grad-CAM
    with col_left:
        st.markdown('<div class="section-header">Grad-CAM Attribution</div>', unsafe_allow_html=True)
        if result.heatmap is not None:
            st.image(
                plot_heatmap_overlay(mel_spec, result.heatmap, "Grad-CAM Overlay"),
                use_container_width=True,
            )
            st.markdown(
                f'<div style="font-size:0.78rem;color:{PALETTE["text_muted"]};margin-top:0.3rem;">'
                "Grad-CAM highlights which time-frequency regions activated the last conv block "
                "for the predicted class.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Grad-CAM not available (model hooks not attached).")

    # SHAP
    band_importance = None
    with col_right:
        st.markdown('<div class="section-header">SHAP Attribution</div>', unsafe_allow_html=True)
        if not SHAP_AVAILABLE:
            st.warning(
                "SHAP not installed. Run:\n```\npip install shap>=0.42.0\n```"
            )
        else:
            with st.spinner("Computing SHAP values…"):
                shap_map, band_importance = compute_shap_heatmap_fast(
                    model, input_tensor, result.label_index, n_background=8
                )
            if shap_map is not None:
                st.image(
                    plot_heatmap_overlay(mel_spec, shap_map, "SHAP Overlay"),
                    use_container_width=True,
                )
                st.markdown(
                    f'<div style="font-size:0.78rem;color:{PALETTE["text_muted"]};margin-top:0.3rem;">'
                    "SHAP GradientExplainer: each pixel's expected marginal contribution to the "
                    f"<b>{result.label}</b> prediction.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("SHAP computation failed for this input.")
                band_importance = None

    # SHAP Frequency Band Importance
    if SHAP_AVAILABLE and band_importance is not None:
        st.markdown('<div class="section-header">Frequency Band Importance (SHAP)</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plotly_shap_bars(band_importance),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            f'<div style="font-size:0.78rem;color:{PALETTE["text_muted"]};">'
            "Mean absolute SHAP value per frequency band. Higher → band drove the prediction more. "
            "Low: 50–500 Hz | Mid: 500–2 kHz | High: 2–8 kHz</div>",
            unsafe_allow_html=True,
        )


# Tab 3: Robustness
def render_tab_robustness(pipeline, waveform, mel_spec, result):
    if result is None:
        st.info("Upload audio in the Analysis tab first.")
        return

    st.markdown('<div class="section-header">SNR Sweep — Confidence Degradation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.82rem;color:{PALETTE["text_muted"]};margin-bottom:1rem;">'
        "Evaluate how the model's confidence and prediction label change as Gaussian noise is added "
        "at different signal-to-noise ratios. Low SNR = more noise.</div>",
        unsafe_allow_html=True,
    )

    snr_levels = [20, 15, 10, 5, 0, -5]
    if "snr_sweep" not in st.session_state:
        st.session_state.snr_sweep = None

    if st.button("Run SNR Sweep", type="primary"):
        confidences, labels, shap_ssims = [], [], []
        prog = st.progress(0, text="Running SNR sweep…")
        for i, snr in enumerate(snr_levels):
            noise   = torch.randn_like(waveform)
            noisy   = mix_at_snr(waveform, noise, float(snr))
            r_noisy = pipeline.predict(noisy, run_quality_checks=False)
            confidences.append(r_noisy.confidence)
            labels.append(r_noisy.label)
            # SHAP stability (SSIM between clean and noisy SHAP maps)
            if SHAP_AVAILABLE and result.heatmap is not None:
                extractor = MelSpectrogramExtractor(sample_rate=16000)
                wav_fixed_noisy = pad_or_truncate(noisy, 16000 * 5)
                mel_n = extractor.extract(wav_fixed_noisy).unsqueeze(0)
                shap_n, _ = compute_shap_heatmap_fast(
                    pipeline.model, mel_n, result.label_index, n_background=5
                )
                wav_fixed_clean = pad_or_truncate(waveform, 16000 * 5)
                mel_c = extractor.extract(wav_fixed_clean).unsqueeze(0)
                shap_c, _ = compute_shap_heatmap_fast(
                    pipeline.model, mel_c, result.label_index, n_background=5
                )
                if shap_n is not None and shap_c is not None:
                    shap_ssims.append(ssim_2d(shap_c, shap_n))
                else:
                    shap_ssims.append(None)
            else:
                shap_ssims.append(None)
            prog.progress((i + 1) / len(snr_levels), text=f"SNR {snr} dB…")
        prog.empty()
        st.session_state.snr_sweep = {
            "snr_levels": snr_levels,
            "confidences": confidences,
            "labels": labels,
            "shap_ssims": shap_ssims,
        }

    if st.session_state.snr_sweep:
        sw = st.session_state.snr_sweep
        st.plotly_chart(
            plotly_snr_curve(sw["snr_levels"], sw["confidences"], sw["labels"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Results table
        st.markdown('<div class="section-header">Per-SNR Results</div>', unsafe_allow_html=True)
        cols = st.columns(len(snr_levels))
        for col, snr, conf, lbl in zip(cols, sw["snr_levels"], sw["confidences"], sw["labels"]):
            changed = lbl != result.label
            label_color = PALETTE["danger"] if changed else PALETTE["success"]
            col.markdown(
                f'<div class="metric-card" style="text-align:center;">'
                f'<div class="metric-label">{snr} dB</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{label_color};">{lbl}</div>'
                f'<div class="metric-sub">{conf:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # JRI computation
        st.markdown('<div class="section-header">Joint Reliability Index</div>', unsafe_allow_html=True)
        valid_ssims = [s for s in sw["shap_ssims"] if s is not None]
        mean_stability = float(np.mean(valid_ssims)) if valid_ssims else 0.5
        clean_conf = result.confidence
        worst_conf = min(sw["confidences"])
        robustness = max(0.0, min(1.0, worst_conf / max(clean_conf, 1e-6)))
        jri = joint_reliability_index(robustness, mean_stability)

        col_jri, col_rob, col_stab = st.columns(3)
        with col_jri:
            st.plotly_chart(
                plotly_jri_gauge(jri),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with col_rob:
            st.markdown(
                f'<div class="metric-card" style="margin-top:1.5rem;">'
                f'<div class="metric-label">Robustness R</div>'
                f'<div class="metric-value">{robustness:.3f}</div>'
                f'<div class="metric-sub">worst_conf / clean_conf</div>'
                f'</div>'
                f'<div class="metric-card">'
                f'<div class="metric-label">SHAP Stability S (mean SSIM)</div>'
                f'<div class="metric-value">{mean_stability:.3f}</div>'
                f'<div class="metric-sub">Structural similarity of SHAP maps under noise</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_stab:
            st.markdown(
                f'<div class="metric-card" style="margin-top:1.5rem;">'
                f'<div class="metric-label">JRI Formula</div>'
                f'<div style="font-size:0.82rem;color:{PALETTE["text"]};line-height:1.7;">'
                "JRI = 2·R·S / (R+S)<br>"
                "<span style='color:{};'>Harmonic mean penalises models that are</span><br>"
                "<span style='color:{};'>robust but produce unstable explanations</span><br>"
                "or stable-but-wrong.".format(PALETTE["text_muted"], PALETTE["text_muted"])
                + "</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div style="font-size:0.85rem;color:{PALETTE["text_muted"]};padding:1rem 0;">'
            "Click <b>Run SNR Sweep</b> to compute the full robustness profile.</div>",
            unsafe_allow_html=True,
        )

    # Interactive single-SNR test
    st.markdown('<div class="section-header">Manual SNR Test</div>', unsafe_allow_html=True)
    snr_db = st.slider("Signal-to-Noise Ratio (dB)", -5, 20, 10, step=1)
    if st.button("Test at this SNR"):
        noise = torch.randn_like(waveform)
        noisy = mix_at_snr(waveform, noise, float(snr_db))
        r_noisy = pipeline.predict(noisy, run_quality_checks=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Clean</div>'
                f'<div class="metric-value">{result.label}</div>'
                f'<div class="metric-sub">{result.confidence:.1%} confidence</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            changed = r_noisy.label != result.label
            color = PALETTE["danger"] if changed else PALETTE["success"]
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Noisy @ {snr_db} dB</div>'
                f'<div class="metric-value" style="color:{color};">{r_noisy.label}</div>'
                f'<div class="metric-sub">{r_noisy.confidence:.1%} confidence'
                f'{"  [changed]" if changed else "  [stable]"}</div></div>',
                unsafe_allow_html=True,
            )
        extractor = MelSpectrogramExtractor(sample_rate=16000)
        wav_n_fixed = pad_or_truncate(noisy, 16000 * 5)
        mel_n = extractor.extract(wav_n_fixed).squeeze().numpy()
        st.image(plot_spectrogram(mel_n, f"Noisy Spectrogram (SNR={snr_db} dB)"),
                 use_container_width=True)


# Tab 4: Model Performance
def render_tab_performance(model, result, input_tensor=None):
    history = load_training_history()
    eval_results = load_eval_results()

    # Training curves
    st.markdown('<div class="section-header">Training History</div>', unsafe_allow_html=True)
    if history:
        st.plotly_chart(
            plotly_training_curves(history),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        best = max(history, key=lambda x: x.get("val_f1_macro", 0))
        st.markdown(
            f'<div style="font-size:0.8rem;color:{PALETTE["text_muted"]};">'
            f'Best epoch: <b>{best["epoch"]}</b> — '
            f'Val F1: <b>{best.get("val_f1_macro", 0):.4f}</b> — '
            f'Train Loss: <b>{best.get("train_loss", 0):.4f}</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No training history found. Run training first:\n\n"
            "```bash\npython -m src.pipeline.train --synthetic --epochs 30\n```"
        )

    # Per-class F1
    if eval_results:
        st.markdown('<div class="section-header">Per-Class F1 Score</div>', unsafe_allow_html=True)
        col_f1, col_summary = st.columns([2, 1])
        with col_f1:
            st.plotly_chart(
                plotly_per_class_f1(eval_results),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with col_summary:
            for k, label in [
                ("f1_macro", "F1"),
                ("accuracy", "Accuracy"),
                ("balanced_accuracy", "Balanced Acc."),
            ]:
                val = eval_results.get(k, 0.0)
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{val:.3f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Faithfulness metrics (only if we have an uploaded file)
    st.markdown('<div class="section-header">Faithfulness Metrics (Live)</div>', unsafe_allow_html=True)
    if result is not None and result.heatmap is not None and input_tensor is not None:
        with st.spinner("Computing faithfulness metrics (Insertion/Deletion AUC, AOPC)…"):
            try:
                hm = result.heatmap
                # Use reduced steps for speed in dashboard
                ins = insertion_auc(model, input_tensor, hm, result.label_index,
                                    n_steps=20, device="cpu")
                dlt = deletion_auc(model, input_tensor, hm, result.label_index,
                                   n_steps=20, device="cpu")
                aop = aopc(model, input_tensor, hm, result.label_index,
                           K=10, device="cpu")
                st.plotly_chart(
                    plotly_faithfulness_bars(ins, dlt, aop),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.markdown(
                    f'<div style="font-size:0.78rem;color:{PALETTE["text_muted"]};">'
                    "<b>Insertion AUC ↑</b>: progressively reveal top-attributed regions from blank — "
                    "higher = explanation captures truly important regions.<br>"
                    "<b>Deletion AUC ↓</b>: progressively remove top-attributed regions — "
                    "lower = removing them drops confidence (faithful).<br>"
                    "<b>AOPC ↑</b>: Area Over Perturbation Curve — average confidence drop after "
                    "removing each top-k feature.</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.warning(f"Faithfulness computation failed: {e}")
    else:
        st.info("Upload audio in the Analysis tab to compute live faithfulness metrics.")

    # Architecture summary
    st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.82rem;color:{PALETTE['text']};line-height:1.8;">
                <b>BaselineCNN</b> — 4-block convolutional backbone with attention pooling<br>
                Channels: 1 → 32 → 64 → 128 → 256 (each block: Conv2d → BN → ReLU → MaxPool)<br>
                Pooling: <b>Attention Pooling</b> over time dimension (learns which time frames matter)<br>
                Head: Linear(2048→128) → ReLU → Dropout(0.3) → Linear(128→4)<br>
                Loss: <b>Focal Loss</b> γ=2.0, α=inverse-class-frequency (handles ICBHI imbalance)<br>
                XAI hooks: Grad-CAM on <code>model.features[-1]</code> (ConvBlock 4)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Main
def main():
    st.set_page_config(
        page_title="RespiScan — Respiratory Sound Analysis",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()

    model, pipeline, is_trained = load_model_and_pipeline()
    render_sidebar(is_trained)

    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:700;color:{PALETTE["text"]};'
        f'margin-bottom:0.1rem;">Respiratory Sound Analysis</div>'
        f'<div style="font-size:0.82rem;color:{PALETTE["text_muted"]};margin-bottom:1.2rem;">'
        f'Noise-Aware Classification · Grad-CAM + SHAP Explainability · Joint Reliability Index</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "  Analysis  ",
        "  Explainability  ",
        "  Robustness  ",
        "  Model Performance  ",
    ])

    # Keep shared state across tabs via session_state
    with tab1:
        waveform, mel_spec, result, extractor = render_tab_analysis(model, pipeline)
        st.session_state["waveform"] = waveform
        st.session_state["mel_spec"] = mel_spec
        st.session_state["result"]   = result

    with tab2:
        render_tab_explainability(
            model,
            st.session_state.get("waveform"),
            st.session_state.get("mel_spec"),
            st.session_state.get("result"),
        )

    with tab3:
        render_tab_robustness(
            pipeline,
            st.session_state.get("waveform"),
            st.session_state.get("mel_spec"),
            st.session_state.get("result"),
        )

    with tab4:
        r = st.session_state.get("result")
        inp_tensor = None
        if r is not None:
            wv = st.session_state.get("waveform")
            if wv is not None:
                ext = MelSpectrogramExtractor(sample_rate=16000)
                wv_f = pad_or_truncate(wv, 16000 * 5)
                inp_tensor = ext.extract(wv_f).unsqueeze(0)
        render_tab_performance(model, r, inp_tensor)


if __name__ == "__main__":
    main()
