from setuptools import setup, find_packages

setup(
    name="PyBioKAN",
    version="0.1.0",
    description="BioKAN - 生体模倣コルモゴロフアーノルドネットワーク",
    author="zapabob",
    author_email="zapabob@gmail.com",
    url="https://github.com/zapabob/biokan",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
        "captum>=0.6.0",
        "plotly>=5.15.0",
        "pillow>=10.0.0",
        "scipy>=1.11.0",
        "optuna>=4.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
) 