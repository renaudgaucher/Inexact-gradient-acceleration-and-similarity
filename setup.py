from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="byzfl",
    version="0.0.1",
    description="Acceleration methods for byzantine fl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["byzfl"],
    python_requires=">=3.8",
)
