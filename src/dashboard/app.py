"""
Streamlit Dashboard for Explainable Respiratory Sound Analysis.

Upload a respiratory audio clip → view prediction, confidence,
Grad-CAM explanation heatmap, and reliability assessment.
Includes interactive SNR simulation slider to explore robustness.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.models.baseline_cnn import BaselineCNN
from src.pipeline.inference import InferencePipeline
from src.features.mel_features import MelSpectrogramExtractor, pad_or_truncate
from src.data.noise_bank import mix_at_snr


def create_model():
    """Create a fresh (untrained) model for demonstration."""
    model = BaselineCNN(num_classes=4, n_mels=128)
    model.eval()
    return model


def plot_spectrogram(waveform, sample_rate=16000, title="Log-Mel Spectrogram"):
    """Plot a log-mel spectrogram from a waveform."""
    extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
    mel = extractor.extract(waveform.squeeze())
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        mel.squeeze().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Mel Bin")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Log Energy")
    plt.tight_layout()
    return fig


def plot_heatmap_overlay(spectrogram, heatmap, title="Grad-CAM Overlay"):
    """Overlay Grad-CAM heatmap on spectrogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Input Spectrogram")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Mel Bin")

    axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma", alpha=0.5)
    axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[1].set_title(title)
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Bin")

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="🫁 Respiratory Sound Analyzer",
        page_icon="🩺",
        layout="wide",
    )

    st.title("🫁 Explainable Respiratory Sound Analysis")
    st.markdown(
        """
        **Noise-Aware Classification & Explainability Validation**

        Upload a respiratory audio recording to analyze it for adventitious sounds
        (crackles, wheezes). The system provides:
        - 🎯 **Classification** with confidence scores
        - 🔍 **Grad-CAM heatmap** showing which time-frequency regions influenced the decision
        - 📊 **Reliability assessment** based on prediction confidence
        - 🔊 **SNR simulation** to explore how noise affects the prediction
        """
    )

    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("---")

    # Model loading
    model = create_model()
    pipeline = InferencePipeline(model, device="cpu", enable_explainability=True)

    st.sidebar.info(
        "⚠️ Using an **untrained** model for demonstration. "
        "Train the model on ICBHI data to get meaningful predictions."
    )

    # File upload
    st.header("📤 Upload Audio")
    uploaded_file = st.file_uploader(
        "Upload a .wav respiratory audio file",
        type=["wav"],
        help="Supported: mono or stereo WAV files at any sample rate.",
    )

    if uploaded_file is not None:
        import torchaudio
        import io

        # Load audio
        audio_bytes = uploaded_file.read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        waveform = waveform.squeeze()

        st.audio(audio_bytes, format="audio/wav")
        st.markdown(
            f"**Duration**: {len(waveform)/sample_rate:.2f}s | "
            f"**Sample Rate**: {sample_rate} Hz | "
            f"**Samples**: {len(waveform):,}"
        )

        # Show spectrogram
        st.subheader("📊 Spectrogram")
        fig = plot_spectrogram(waveform, sample_rate)
        st.pyplot(fig)

        # Run prediction
        st.subheader("🎯 Prediction")
        result = pipeline.predict(waveform, run_quality_checks=True)

        if result.label == "REJECTED":
            st.error(f"❌ Audio rejected: {', '.join(result.quality_report.reasons)}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Class", result.label)
            with col2:
                st.metric("Confidence", f"{result.confidence:.1%}")
            with col3:
                reliability_colors = {
                    "high": "🟢",
                    "medium": "🟡",
                    "low": "🔴",
                }
                icon = reliability_colors.get(result.reliability_flag, "⚪")
                st.metric("Reliability", f"{icon} {result.reliability_flag.upper()}")

            # Class probabilities
            st.subheader("📈 Class Probabilities")
            prob_data = result.class_probabilities
            fig_prob, ax = plt.subplots(figsize=(8, 3))
            colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
            bars = ax.barh(list(prob_data.keys()), list(prob_data.values()), color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Class Probabilities")
            for bar, val in zip(bars, prob_data.values()):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center")
            plt.tight_layout()
            st.pyplot(fig_prob)

            # Grad-CAM heatmap
            if result.heatmap is not None:
                st.subheader("🔍 Explanation (Grad-CAM)")
                target_length = int(sample_rate * 5.0)
                wav_fixed = pad_or_truncate(waveform, target_length)
                extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
                spec = extractor.extract(wav_fixed).squeeze().numpy()
                fig_cam = plot_heatmap_overlay(spec, result.heatmap)
                st.pyplot(fig_cam)

        # SNR Simulation
        st.subheader("🔊 Noise Robustness Simulation")
        st.markdown("Simulate how noise affects the prediction by adjusting the SNR slider.")

        snr_db = st.slider("Signal-to-Noise Ratio (dB)", -5, 20, 10, step=1)

        if st.button("🔊 Simulate Noisy Prediction"):
            # Generate noise and mix
            noise = torch.randn_like(waveform)
            noisy = mix_at_snr(waveform, noise, float(snr_db))

            # Show noisy spectrogram
            fig_noisy = plot_spectrogram(noisy, sample_rate, f"Noisy Spectrogram (SNR={snr_db} dB)")
            st.pyplot(fig_noisy)

            # Predict on noisy audio
            result_noisy = pipeline.predict(noisy, run_quality_checks=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Clean Prediction**")
                st.write(f"Class: {result.label} ({result.confidence:.1%})")
            with col2:
                st.markdown("**Noisy Prediction**")
                st.write(
                    f"Class: {result_noisy.label} ({result_noisy.confidence:.1%})"
                )

            if result.label != result_noisy.label:
                st.warning("⚠️ Noise changed the predicted class!")
            else:
                st.success("✅ Prediction is robust to this noise level.")

    # Quality report
    if uploaded_file is not None and result.quality_report is not None:
        with st.expander("📋 Quality Report"):
            qr = result.quality_report
            st.json(
                {
                    "passed": qr.passed,
                    "reasons": qr.reasons,
                    "duration_sec": round(qr.duration_sec, 3),
                    "peak_amplitude": round(qr.peak_amplitude, 6),
                    "rms_energy": round(qr.rms_energy, 6),
                    "silence_fraction": round(qr.silence_fraction, 4),
                    "clipping_fraction": round(qr.clipping_fraction, 4),
                }
            )

    pipeline.cleanup()
    st.markdown("---")
    st.caption(
        "🩺 **Disclaimer**: This is a research prototype and is NOT a diagnostic tool. "
        "Not intended for clinical use."
    )


if __name__ == "__main__":
    main()
