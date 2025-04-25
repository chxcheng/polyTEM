"""
polyTEM: A Python package for Transmission Electron Microscopy (TEM) image analysis and processing.

This package provides a comprehensive suite of tools for analyzing and processing TEM data, with a particular focus on:
- Crystal structure analysis and peak detection
- Image processing and enhancement
- Spatial analysis of crystalline materials
- Tomographic reconstruction and analysis
- Paracrystalline structure analysis
- Statistical analysis of TEM data

Key features include:
- Crystal peak detection and analysis
- Contrast Transfer Function (CTF) correction
- Drift correction for multi-frame images
- Fourier analysis and bandpass filtering
- Feature extraction for periodic structures
- STEM data processing and analysis
- 3D reconstruction and analysis
- Statistical analysis of crystalline structures

The package is designed to work with both CPU and GPU processing, providing efficient tools for handling large TEM datasets.
"""
__author__ = "Christina Cheng"
__version__ = "v0.2"
__all__ = ['crystal_peaks',
           'image',
           'spatial',
           'tomography',
           'paracrystalline',
           'params',
           'statistics',
           'tol_colors',
           'utilities']

from . import crystal_peaks
from . import image
from . import spatial
from . import tomography
