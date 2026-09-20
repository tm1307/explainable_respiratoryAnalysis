from setuptools import setup, find_packages

setup(
    name="explainable_respiratory_analysis",
    version="0.1.0",
    description="AI-Based Respiratory Sound Screening — Noise-Aware Classification & Explainability Validation",
    author="tm1307",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "matplotlib>=3.7.0",
        "captum>=0.7.0",
        "scikit-image>=0.21.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "hydra-core>=1.3.0",
    ],
)
