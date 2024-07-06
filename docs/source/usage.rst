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
3. **domain** with the *Domain* and *DomainCollection* classes, which handles clustering of crystalline domains
4. **spatial** which handles the spatial and statistical analysis
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


