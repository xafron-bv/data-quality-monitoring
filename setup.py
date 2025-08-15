from setuptools import setup, find_packages

setup(
    name="data-quality-monitoring",
    version="0.1.0",
    description="Data Quality Monitoring System",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
    ],
)