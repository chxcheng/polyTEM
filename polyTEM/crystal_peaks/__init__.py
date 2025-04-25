"""
Crystal Peaks subpackage for analyzing crystalline materials from diffraction data.

This subpackage provides functionality for:
- Processing and analyzing diffraction patterns from crystalline materials
- Extracting and analyzing crystal orientations and peaks
- Generating flow fields and director fields from crystal orientation data
- 3D crystal stack analysis and visualization
- Conditional probability analysis of crystal orientations
- Vector graph utilities for crystal orientation networks
"""

__all__ = ['_conditional_probability', '_process_datacube','crystal_stack','flow_fields','vector_graph_utils']

from .crystal_stack import CrystalStack
from .crystal_stack_3D import CrystalStack3D