from . import fft
from ..crystal_peaks import crystal_stack as peaks

import logging
from scipy.stats import multivariate_normal
import torch
import numpy as np

class SED:
    """
    Scanning Electron Diffraction
    
    This class is designed to work with the output .npy files after pre-processing
    of 4DSTEM data either using py4Dstem or the Salleo Group code written by Luke Balhorn.
    Thus the input data is assumed to be of shape (position, q, angles). The positions are 
    the flattened index from the 2D raster scan of the sample area, which is assumed to be 
    perfectly square.
    
    Attributes:
        data (array-like): with shape (position, q, angle)
        data_params (dict): 4D-STEM Data Acquisition Parameters. Expected Keys
            "q_min", "q_max", "q_step", "probe_step" (nm), "angle_step"
        process_params (dict): Data Analysis Parameters. Expected Keys
            "peak_q","peak_q_sigma", "peak_theta_sigma", "processor"
        bkg (np.ndarray): shape (images, q). Median intensity for each q in 
            each diffraction pattern
        
    """
    required_params = ['q_min','q_max','q_step','probe_step','angle_step']
    
    def __init__(self, data, data_params, process_params):
        self.data = data
        if all(par in data_params.keys() for par in self.required_params):
            self.data_params = data_params
            self.q = np.linspace(data_params["q_min"],
                                 data_params["q_max"], 
                                 int(np.round((data_params["q_max"] - data_params["q_min"])/data_params["q_step"])),
                                 endpoint=False)
            self.theta = np.arange(0,180,data_params["angle_step"])
        else:
            logging.error("Not all required parameters were provided! Some functions may fail")
            
        self.process_params = process_params
        
        # The median intensity for each q in each diffraction pattern
        # This is an estimate of the background noise for each q
        self.bkg = np.median(self.data,axis=2)[:,:,None]

        return
    
    def _orientation_filters(self):
        """ 
        Create set of gaussian filters to extract diffraction peak intensity
        as function of angle, discretized based on data's angle_step.
        
        Returns:
            gaussian_filters (np.ndarray): Shape (q, theta, theta)
        """
        num_pos, num_q, num_th = self.data.shape
        xx, yy = np.meshgrid(self.theta,self.q)
        xy = np.dstack((xx,yy))
        gaussian_filters = np.zeros(shape=(num_q,num_th,num_th))
        
        # Create multivariable gaussian centered at (q,th), for range of th
        for th_index,th_mean in enumerate(np.arange(0,180,self.data_params["angle_step"])):
            rv = multivariate_normal(mean = [th_mean, self.process_params['peak_q']], 
                                     cov = [self.process_params["peak_theta_sigma"],
                                            self.process_params["peak_q_sigma"]]
                                    )                               
            gaussian_filters[:,:,th_index] = rv.pdf(xy)
        return gaussian_filters
    
    def featurize(self, peak_threshold_function, verbose=False):
        """
        Extract local orientation from peak features in data
        """
        num_pos, num_q, num_th = self.data.shape
        # diffraction pattern minus the background
        peak_data = self.data - self.bkg
        
        intensity_theta = torch.empty(size=(num_pos,num_th))
        data = torch.Tensor(peak_data)\
                    .to(self.process_params["processor"])\
                    .double()
        orientation_filters = torch.Tensor(self._orientation_filters())\
                                   .to(self.process_params["processor"])\
                                   .double()
        for i in range(num_pos):
            intensity_theta[i,:] = torch.sum(
                                        torch.mul(data[i,:,:,None], orientation_filters),
                                        [0,1])
        
        # reshape intensity_theta into (x,y,th)
        # The assumption is the data was rastered in a square
        num_rows = int(np.round(num_pos**0.5))
        datacube = intensity_theta.reshape((num_rows, num_rows, num_th))
        
        # CrystalStack requires that datacube has angle resolution of 1 degree
        # extend and take average to interpolate 
        datacube_180 = np.repeat(datacube,self.data_params["angle_step"],axis=2)
        datacube = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(self.data_params["angle_step"]),'same')/self.data_params["angle_step"],
            axis = 2,
            arr = datacube_180)
        
        # Get CrystalStack object with local orientation
        crystalStack = peaks.CrystalStack(img=None,
                                          datacube=datacube, 
                                          peak_threshold_function=peak_threshold_function)
        crystalStack.resolution = self.data_params["probe_step"]
        crystalStack.q = self.process_params["peak_q"]
        crystalStack.peaks_from_datacube(verbose=verbose)
        
        return crystalStack
    
        