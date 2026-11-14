from setuptools import find_packages, setup

setup(
    name="tribe_local",
    version="0.1.0",
    description="Lightweight GPU-Compatible Multimodal Brain Encoder inspired by Meta TRIBE v2",
    author="Independent Research Implementation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "pytorch-lightning>=2.2.0",
        "einops>=0.7.0",
        "openai>=1.20.0",
        "tenacity>=8.2.0",
        "pyyaml>=6.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "fmri": ["nibabel>=5.1.0", "nilearn>=0.10.0", "h5py>=3.10.0"],
        "dev": ["pytest", "black", "isort"],
    },
)
