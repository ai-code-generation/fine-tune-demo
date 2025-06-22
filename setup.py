#!/usr/bin/env python3
"""
Setup script for the NeMo Fine-tuning Pipeline.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nemo-finetuning-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive fine-tuning pipeline for CodeLlama and Llama3 models using NeMo Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nemo-finetuning-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "flash-attn": [
            "flash-attn>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nemo-finetune=finetune_pipeline:main",
            "nemo-evaluate=evaluate_model:main",
            "nemo-preprocess=data_preprocessing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md", "*.txt"],
    },
)
