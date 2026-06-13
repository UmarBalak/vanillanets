"""
Setup configuration for VanillaNets package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vanillanets",
    version="1.0.0",
    author="Umar Balak",
    author_email="umarbalak35@gmail.com",
    description="A transparent, NumPy-only neural network library for learning and experimentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UmarBalak/vanillanets",
    project_urls={
        "Bug Tracker": "https://github.com/UmarBalak/vanillanets/issues",
        "Documentation": "https://github.com/UmarBalak/vanillanets#readme",
        "Source Code": "https://github.com/UmarBalak/vanillanets",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    keywords="neural-network deep-learning machine-learning numpy education",
)
