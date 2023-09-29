"""Install script for setuptools."""
from __future__ import absolute_import, division, print_function

from setuptools import setup

setup(
    name="psro",
    version="0.0.2",
    description="Policy-Space Resposne Oracles building blocks.",
    author="Max Smith",
    packages=["psro"],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.7.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
