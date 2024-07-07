Usage
=====

.. note::

   This project is under active development. Documentation is publicly
   available to demonstrate the working applications of this package. 
   Version-controlled source code will be released as manuscripts are published.


There are five main modules:

1. **image** with the *Image* class, which handles the construction and image post-processing of a single HRTEM micrograph
2. **crystal_peaks** with the *CrystalStack* and *CrystalStack3D* classes, which handle the analysis of crystalline signal 
   extracted by Fourier analysis of an Image.
4. **spatial** which handles the spatial and statistical analysis, including domain clustering
5. **tomography** which handles tilt series alignment and reconstruction

image module
-------------
The *Image* class is useful for loading, viewing, and post-processing the HRTEM micrograph 
(note: does not contain analysis). The main methods include:

* CTF correction
   * CTF correction is handled by the ctf submodule and *CTF* class, which takes the .txt output of the CTFFIND4_ software
* Frame alignment (for dose-fractionated images)
* Applying masks and bandpass filters 
* Computing the FFT and power spectrum
   * Computing a sliding FFT scan will also output a *CrystalStack* object to enable further analysis.

.. _CTFFIND4: https://grigoriefflab.umassmed.edu/ctffind4/

crystal_peaks module
--------------------
*CrystalStack* class handles the scanning fft peak results from a single projection image, 
while *CrystalStack3D* handles a list of *CrystalStacks* for 3D analysis to produce a networkx_
graph of connected backbone nematic directors. 

*CrystalStack*
   * Finds peaks in scanning FFT windows, stored as one-hot sparse_ matrix
   * Distribution of overlap angles
   * Conditional probability of backbone misorientation given separation distance
   * (v0.1) Handles domain clustering
   * 2D backbone flowline visualization is handled by the **flow_fields** submodule.

.. _sparse: https://sparse.pydata.org/en/stable/


*CrystalStack3D*
   * Creates 3D nematic director fields 
   * Aligns director fields based on *AlignmentResult*
   * Constructs networkx_ graph representation of 3D connected backbone directors
      * Calculates adjacency probability matrix from bending probability 
      * Constructs graph by sampling the adjacency probability matrix
   * For 3D graph visualization, see :doc:'3Dviz.rst'

.. _networkx: https://networkx.org/documentation/stable/reference/introduction.html


spatial module (v0.2)
---------------------
Crystalline domains are clustered using HDBSCAN_ which outputs a *DomainCollection* object, whose *domains* attribute
is a list of *Domain* objects for each cluster. *Domain* geometries are represented using alphashape_ and shapely polygons_. 
*DomainCollections* can then be analyzed for

* Orientation correlation length
* Size and Shape distribution

The **stats** module handles statistical methods including:

* Extrapolations and Interpolations
* Model fitting 
   * 2nd Legendre Polynomial
   * exponential decay
   * Frank-Oseen's bending probability
* Hypothesis Testing
   * Z-test and Fisher's Z-test
   * Kolmogorov-Smirnov test
* Statistical analysis  
   * Auto-correlation and cross-correlation
   * Spatial autocorrelation (Global and local Moran's I)

Version Release Notes: In v0.1 these functions were contained in separate modules. 
In v0.2, they will all be handled by the main spatial module. Code is still being migrated and refactored.

.. _HDBSCAN: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
.. _alphashape: https://pypi.org/project/alphashape/
.. _polygons: https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html

tomography module (v0.2)
------------------------
The *AlignmentResult* class handles alignment of tilt series image frames using fiduciary marker alignment
based on Jing and Sachs, [JingSachs]_. 

* Marker locations can be manually inputted or found using unsupervised clustering
* Calculates alignment error (rotation and displacement)
* *chain_affine_transformation_mats* provides transformation between original image frame
   coordinates to aligned projection coordinates
* aligned projection coordinates are mapped to sample coordinates using *proj_tilt_mat* inverse

.. [JingSachs] Jing ZQ, Sachs F. 
   Alignment of tomographic projections using an incomplete set of fiducial markers. 
   Ultramicroscopy. 1991 Jan;35(1):37-43. doi: 10.1016/0304-3991(91)90042-5. PMID: 2063493.

Unit-cell reconstruction, adapted from single-particle electron tomography, using weighted
back-projection and filtered back-projection are still in development.

