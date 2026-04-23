from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="supernode-tokenizer",
    version="0.1.0",
    description="Standalone RLBench imitation learning codebase for point-cloud tokenizer comparisons.",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "absl-py>=2.1",
        "diffusers>=0.30",
        "h5py>=3.10",
        "ml-collections>=0.1.1",
        "numpy>=1.24",
        "Pillow>=10.0",
        "torch>=2.1",
        "tqdm>=4.66",
    ],
    extras_require={
        "video": ["imageio>=2.34"],
        "wandb": ["wandb>=0.17"],
    },
)
