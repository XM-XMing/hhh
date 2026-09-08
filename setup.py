#!/usr/bin/env python3
"""Catkin-compatible installation for the planning Python package."""

from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages, setup


setup_args = generate_distutils_setup(
    packages=find_packages("python"),
    package_dir={"": "python"},
)

setup(**setup_args)
