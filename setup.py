# setup.py
from setuptools import setup, find_packages

setup(
    name="mafsurv",
    version="1.0.0",
    description="MAF-Surv: Enhanced Cancer Survival PredictionFramework with Multimodal Data Fusion",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "lifelines>=0.26.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
