# Lint as: python3

import os

from setuptools import find_packages, setup

REQUIRED_PKGS = [
    "Pillow==9.0.1",
    "pandas",
    "matplotlib==3.5.1",
    "Jinja2==3.0.3",
    "python-Levenshtein==0.12.2",
    "opencv-python==4.6.0.66",
    "scipy==1.7.3"
]

setup(
    name="iliauniiccocrevaluation",
    version="1.0.0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Iliauni ICC open source OCR evaluation library",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="IliaUni ICC",
    author_email="anzor.gozalishvili@gmail.com",
    url="https://github.com/IliaUni-ICC/ocr_evaluation",
    download_url="https://github.com/IliaUni-ICC/ocr_evaluation/tags",
    license="Apache 2.0",
    package_dir={"": "ocr_evaluation"},
    packages=["ocr_evaluation"],
    entry_points={},
    install_requires=REQUIRED_PKGS,
    extras_require={},
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="metrics machine learning evaluate evaluation ocr",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
