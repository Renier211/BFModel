from setuptools import setup, find_packages

setup(
    name="BFModel",                      # Package name
    version="0.1",                       # Version
    packages=find_packages(),            # Automatically include all sub-packages
    description="A package for BFModel", # Short description
    install_requires=[                   # List dependencies here
        "numpy",
        "pandas",
    ],
    python_requires=">=3.6",             # Minimum Python version requirement
)
