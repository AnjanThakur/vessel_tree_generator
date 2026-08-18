"""Person 2 synthetic coronary-tree generation package.

The package consumes, but does not modify, the Person 1 ``surface_relative``
API and frozen statistical artifacts.
"""

from __future__ import annotations

from .landmark_sampler import LandmarkSampler, SurfaceLandmark
from .parameter_sampler import EllipsoidParameters, ParameterSampler
from .tree_assembler import SyntheticTree, TreeAssembler

__all__ = [
    "EllipsoidParameters",
    "LandmarkSampler",
    "ParameterSampler",
    "SurfaceLandmark",
    "SyntheticTree",
    "TreeAssembler",
]
