"""
The image subpackage provides comprehensive tools for processing and analyzing TEM (Transmission Electron Microscopy) images.

This subpackage contains several specialized modules for different aspects of TEM image processing:

- Image.py: Core class for handling HRTEM images with methods for loading, saving, and basic image processing
- ctf.py: Tools for Contrast Transfer Function (CTF) correction and analysis
- drift.py: Functions for correcting drift in multi-frame TEM images
- fft.py: Fourier transform operations and frequency domain analysis
- stem.py: Scanning Electron Diffraction (SED) analysis tools

Key features include:
- Support for both CPU and GPU processing
- CTF correction and analysis
- Drift correction for multi-frame images
- Fourier analysis and bandpass filtering
- Feature extraction for periodic structures
- STEM data processing and analysis

The modules are designed to work together seamlessly, with the Image class serving as the main interface
for most operations. The package supports various image formats including MRC, NPY, and standard image formats.
"""

__all__ = ['ctf', 'drift', 'fft', 'Image','stem']

from .Image import Image
from .stem import SED