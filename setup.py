# read the contents of your README file
from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pyCHIPS",
    version="0.4.5",
    packages=["chips"],
    package_dir={"chips": "chips"},
    package_data={"chips": []},
    author="Shibaji Chakraborty",
    author_email="shibaji7@vt.edu",
    maintainer="Shibaji Chakraborty",
    maintainer_email="shibaji7@vt.edu",
    license="MIT License",
    description="CHIPS: Coronal Hole Identification using Probabilistic Scheme",
    long_description="Identify coronal holes and coronal hole boundaries using probabilistic schemes",
    install_requires=[
        "matplotlib>=3.3.2",
        "loguru",
        "sunpy",
        "aiapy",
    ],
    keywords=[
        "python",
        "coronal hole",
        "coronal hole boundary",
        "coronal hole identification",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/shibaji7/pyCHIPS",
)
