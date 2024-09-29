# import Nano
from . import _process_datacube, _conditional_probability, flow_fields
from .. import utilities
import sparse
import torch
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import time
import hdbscan
import pandas as pd
import dill 
import logging
import os
import jsonpickle.ext.numpy as jsonpickle_numpy

"""
Crystalline signal obtained from FFT peaks
"""

class MultiCrystalStack:

    def __init__(self):
        return
    
    def compare_theta(stackA, stackB, save = False, savevar = 'delta_th', outfile = '', plot = True):
        '''
        Compares the local orientation between two different Crystal Stack peaks
        '''
        # first set up average comparison by clusters
        # check if peaks df
        if stackA.peaks_df is None or stackB.peaks_df is None:
            print('Error: Run CrystalStack.get_clusters() first!')
            
        overlap_df = pd.merge(stackA.peaks_df, stackB.peaks_df, on=['coo'], how='inner')
        overlap_df['delta_th'] = overlap_df['theta_x'] - overlap_df['theta_y']
        overlap_df['distance_th'] = [peak_distance(x,y) for x,y in zip(overlap_df['theta_x'],overlap_df['theta_y'])]
        if plot:
            fig, axs = plt.subplots(1,2)
            hist = sns.histplot(data = overlap_df, x = 'delta_th', kde = True, binwidth = 1, ax = axs[0])
            sns.histplot(data = overlap_df, x='distance_th', kde=True, ax = axs[1])
            plt.show()
        
        if save and outfile:
            # save the histogram counts to be binned with other samples
            overlap_df[savevar].value_counts().reset_index().to_csv(outfile)
        return overlap_df

def _threshold_90th(x):
        return np.percentile(x,90)
    
def _threshold_1sigma(x):
        return np.min(x)+np.std(x)
    
class CrystalStack:
    """
    FFT peaks from HRTEM Image
    
    Attributes:
        q (float): 1/A
        resolution (float): nm/pxel
        outdir (str): save directory
        hkl (int): 100, 010, 001 for lamellar, pi-pi, or backbone information
        datacube (np.ndarray): sliding fft scan intensities
        type_: default 'Donor'
        sparse_peaks_mat (sparse): one-hot matrix of location and orientation 
        conditional_probability (ConditionalProbability): class containing Probability of orientation difference given distance
        ks_distmat:
        ks_dlist:
        ks_thetas:
        peaks_df (pd.Dataframe):
        _threshold_function : _threshold_1sigma
        _plot_frequency : 20000
        _peak_width_threshold : 5
    """

    def __init__(self, img, datacube, hkl = 100, type_ = 'Donor', peak_threshold_function = _threshold_1sigma):

        # Img is of nanoclass
        ####
        # Create Attributes
        if img is not None:
            self.q = img.q_center
            self.resolution = img.pixel_size_after_reduction #in nm 
            self.outdir = img.output_folder
        else:
            self.q = None
            self.resolution = None # nm to pxl of datacube
            self.outdir = None

        self.hkl = hkl
        self.datacube = datacube
        self.type_ = type_
        self.sparse_peaks_mat = None
        self.conditional_probability = None # Matrix of conditional probabilities for orientation difference given distance
        self.ks_distmat = None # for Kolmogorov-Smirnov Test against randomly oriented, distribution matrix
        self.ks_dlist = None # For Kolmogorov- Smirnov Test against randomly oriented, D-value list
        self.ks_thetas = None # For Kolmogorov-Smirnov Test against randomly oriented, list of theta with max difference for each distance
        self.peaks_df = None # columns = coo_x, coo_y, theta, sin_orientation, complex_theta, cluster_num, confidence, coo

        self._threshold_function = peak_threshold_function
        self._plot_frequency = 20000
        self._peak_width_threshold = 5
        
        if self.datacube.shape[2] != 180:
            logging.warning(f"Datacube does not have full angle resolution. datacube has {datacube.shape[2]} angles, but expected 180")
        return
    

    def peaks_from_datacube(self, save = False, plot_overlap_angles = True, show_figures = True, verbose=False, plot_freq=100):
        """
        Find fft peaks in datacube and records peak location, 
        
        saved to CrystalStack.sparse_peaks_mat as a sparse one-hot matrix of (x,y,theta)
        """

        if torch.is_tensor(self.datacube):
            if self.datacube.is_cuda:
                self.datacube = self.datacube.to('cpu')
            self.datacube = self.datacube.numpy()

        peaks_matrix, self.overlap_angles = _process_datacube._find_datacube_peaks(self.datacube, self._threshold_function,
         width=self._peak_width_threshold, plot_freq=plot_freq, verbose=verbose)

        if save:
            self.save_peaks()

        # Average number of peaks
        nonzero = np.count_nonzero(peaks_matrix, axis=(2))
        print('     ...Average number of peaks per grid point: ', np.mean(nonzero))
        print('     ...Maximum number of peaks per grid point: ', np.max(nonzero))

        if plot_overlap_angles:
            self.plot_overlap_angles()

        self.sparse_peaks_mat = sparse.COO.from_numpy(peaks_matrix)
        return 

    @classmethod
    def load(cls,filename):
        """
        Load saved crystalstack

        Args:
            filename (str): ends in extension .str or .pkl
        Returns:
            CrystalStack object
        """
        ext = os.path.splitext(filename)[1]
        if ext == '.json':
            stack = utilities.load_json(filename)
        else:
            with open(filename,'rb') as inp:
                stack = dill.load(inp)
        return stack
    
    def save(self,prefix='',json=False):
        """
        Args:
            prefix (str): save file "[prefix]crystal_stack"
            json (bool): if True, save as .json, else save as .pkl
        """
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        filename = os.path.join(self.outdir,f'{prefix}crystal_stack')
        if json:
            jsonpickle_numpy.register_handlers()
            utilities.saveas_json(self,f'{filename}.json')
        else:
            with open(os.path.join(self.outdir,f'{filename}.pkl'), 'wb') as outp:
                dill.dump(self, outp)
        return

    def plot_overlap_angles(self):
        sns.histplot(self.overlap_angles, kde=True, bins=len(set(self.overlap_angles)), color='darkblue',
                     line_kws={'color': 'black', 'linewidth': 0.5})
        plt.xlim([0, 90])
        plt.xlabel('Relative overlap angle / degrees', fontsize=14)
        plt.ylabel('Overlap probability', fontsize=14)

        plt.show()
        plt.close()
        return

    def conditional_prob(self, threads=8, num_slices = 16, calculate=False, plot=True,**kwargs):
        """
        Calculate and/or plot Conditional Probability of misorientation given spatial distance R
        
        Args:
            threads: int, number of multiprocessing cores to use
            plot: bool, whether to plot
            **kwargs: optional plotting keyword-arguments
            plot_color:string
            plot_dist_max: int
            plot_th_max: int
        
        """
        if self.conditional_probability is None or calculate is True:
            self.conditional_probability = _conditional_probability.get_cond_prob(self.sparse_peaks_mat, plot,
                                                                                  num_slices = num_slices,
                                                                   resolution=self.resolution, threads=threads)
        if plot:
            palette=kwargs.get('plot_color','reds')
            max_d=kwargs.get('plot_dist_max',25)
            max_dtheta = kwargs.get('plot_th_max',20) 
            self.conditional_probability.plot(colors=palette,xmax=max_d,ymax=max_dtheta)
        return

    def get_clusters(self, min_cluster_size=10, min_samples=5, cluster_selection_epsilon=10, match_reference_implementation=False, plot = True):
        """
        Cluster fft peaks to find crystalline domains using HDBSCAN

        Saves cluster information as peaks_df attribute. 
        see HDBSCAN documentation for more description on parameters.

        Args:
            min_cluster_size (int):
            min_samples (int): in order to be considered a cluster
            cluster_selection_epsilon:
            match_reference_implementation (bool):
            plot (bool)
        
        """
        _clusterer, self.peaks_df = _get_clusters(self.sparse_peaks_mat, self.resolution,
                                                  min_cluster_size=min_cluster_size, 
                                                  min_samples=min_samples, 
                                                  cluster_selection_epsilon=cluster_selection_epsilon,
                                                  match_reference_implementation=match_reference_implementation)

        if plot:
            self.plot_clusters()

    def plot_clusters(self, confidence_threshold=0.75, probabilities=False, orientations=True, color = None):
        '''
        Attention:
            DEPRECATE IN THE FUTURE. PREFERENCE FOR spatial.plot_polygons_df
        '''
        logging.warning('Soon to be deprecated in favor of spatial.plot_polygons_df(cluster_df)')
        
        # filter out based on confidence threshold
        plot_data = self.peaks_df[self.peaks_df['confidence'] > confidence_threshold]
        if orientations:
            # color based on median orientation
            color_palette = sns.color_palette('husl', 180)
            med_orientations = plot_data.groupby("cluster_num")["theta"].median()
            uniq_orientations = np.unique(plot_data["theta"])
#             color_indices = [np.argmin(np.abs(uniq_orientations - med_orientations[x]))
#                               for x in plot_data["cluster_num"]]
            color_indices = [int(med_orientations[x]) for x in plot_data["cluster_num"]]
            cluster_colors = [color_palette[index] for index in color_indices]
            print("Colored by Orientation")
        elif color is not None:
            cluster_colors = [color for x in plot_data['cluster_num']]
        else:
            # color by cluster number
            color_palette = sns.color_palette('tab10', plot_data["cluster_num"].nunique())
            cluster_colors = [color_palette[x] for x in plot_data["cluster_num"] if x >= 0]
            if probabilities:
                cluster_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, peaks_df["probabilities"])]

        # create plot
        fig = plt.figure(figsize = (10,10))
        plt.scatter(plot_data.coo_y, plot_data.coo_x, s = 6, linewidth=0, c=cluster_colors, alpha = 1)
        plt.title("Cluster Map")
        plt.xlabel("X (pixel)")
        plt.ylabel("Y (pixel)")
        plt.gca().invert_yaxis()
        if orientations:
            cmap = mpl.colors.ListedColormap(sns.color_palette('husl',180))
            norm = plt.Normalize(0, 180)
            sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
            plt.colorbar(sm, label='Orientation (deg)')

            plt.title("Cluster Map by Orientation")
        plt.show()
        return fig
       
    def compare_to_random(self, reference='uniform', rand_stack=None, plot = True):
        if self.conditional_probability is None:
            print('Error: Please get conditional probabilities first.')
        elif reference == "discrete uniform" and rand_stack is None:
            rand_stack = self.random(n=self.sparse_peaks_mat.nnz, dim=self.sparse_peaks_mat.shape, img=None)
            rand_stack.conditional_prob(plot=False)
            
        self.ks_distmat, self.ks_dlist, self.ks_thetas = _conditional_probability.compare_to_random(self.conditional_probability, reference, rand_stack, plot) 
        return

    @classmethod
    def random(cls, n, dim, img):
        """
        Generates a crystal stack with randomized peaks matrix,
        effectively generates a crystal stack with random orientation
        """
        rand_stack = cls(img, datacube=None, type_="Random")
        peaks_mat = _process_datacube._random(n, shape=dim)
        rand_stack.sparse_peaks_mat = sparse.COO.from_numpy(peaks_mat)
        return rand_stack
    
    def apply_mask(self,mask, inplace=True):
        """
        Apply (x,y) mask
        """
        #reshape mask
        mask = np.repeat(mask[:,:,np.newaxis],180,axis=2)
        if inplace:
            self.sparse_peaks_mat *= mask
            return
        else:
            return self.sparse_peaks_mat * mask
        
    def plot_flow_field(self, perpendicular=True, seed_density=2, min_length=5, bend_tolerance=5, curve_resolution=1, preview_sparsity=20, line_spacing=1, line_width_scale=2, spacing_resolution=5, angle_spacing_degrees=10, max_overlap_fraction=0.5, inverse_colors=True, contrast = 1, gamma = 0.1, show_preview=False, show_figures=True, save_results=False, output_folder=''):
        """
        Plot Flow Fields
        """
        flow_maps = flow_fields.plot_flow_fields(self.datacube,
                                     self.sparse_peaks_mat.todense(),
                                     step_size = self.resolution,
                                     perpendicular=perpendicular,
                                     seed_density=seed_density,
                                     bend_tolerance=bend_tolerance,
                                     min_length=min_length,
                                     curve_resolution=curve_resolution,
                                     preview_sparsity=preview_sparsity,
                                     line_spacing=line_spacing,
                                     line_width_scale=line_width_scale,
                                     spacing_resolution=spacing_resolution,
                                     angle_spacing_degrees=angle_spacing_degrees,
                                     max_overlap_fraction=max_overlap_fraction,
                                     inverse_colors=inverse_colors,
                                     contrast=contrast,
                                     gamma=gamma,
                                     show_preview=show_preview,
                                     show_figures=show_figures,
                                     save_results=save_results,
                                     output_folder=output_folder)
        return flow_maps


    def get_orientation_correlation_length(self,min_dist=0,max_dist=20, plot=True):
        """
        Calculates the orientation correlation length given by the exponential decay fit of
        the expected value <2cos^2(\Delta\theta(r))-1>
        
        Returns:
            Ld (float): Length in nanometers
            model (named tuple): LinRegressModel(model.slope,model.intercept,model.rvalue)
        """  
        from ..spatial import stats
        ev = self.conditional_probability.expected_value(stats.legendre_polynomial_2(2))
        x = self.conditional_probability.d
        Ld, model = stats.exponential_decay_length(y=ev, x=x, start_ind=min_dist, end_ind=max_dist)
        
        self.orientation_correlation_length = Ld*self.resolution
        
        if plot:
            plt.figure()
            plt.plot(x, np.log(ev),label='raw')
            plt.plot(x, model.intercept + x*model.slope,'r--',label='fit')
            plt.xlabel('Distance (pixels)')
            plt.ylabel(r'$\ln(\langle 2\cos^2\theta - 1\rangle)$')
        return self.orientation_correlation_length, model
    



#####################################
def _threshold_standard(x):
        return np.percentile(x,90)
    
def _stack_from_file(filename):
    return CrystalStack.stack_from_file(filename)
    
def _get_clusters(sparse_peaks_mat, nm_per_pxl,min_cluster_size = 10, min_samples = 5, cluster_selection_epsilon=10,match_reference_implementation=True):
    ''' 
    uses HDBSCAN clustering algorithm to find clusters
    Args:
        sparse_peaks_mat = one-hot sparse matrix that encodes each peak
        nm_per_pxl = resolution of sparse_peaks_mat
        min_cluster_size = scalar parameter HDBSCAN
        min_samples = scalar parameter for HDBSCAN
        cluster_selection_epsilon: scalar parameter for HDBSCAN
        match_reference_implementation: boolean parameter for HDBSCAN
    
    OUTPUTS:
    clusterer = output of HDBSCAN algorithm
    peaks_df = pandas dataframe containing peak and cluster information
    '''
    start_time = time.time()
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                               cluster_selection_epsilon=cluster_selection_epsilon,
                               match_reference_implementation=match_reference_implementation)

    # convert theta into husl colorwheel values to maintain cyclic nature
    color_palette = sns.color_palette('husl', 180, as_cmap=True)
    cpal = plt.cm.ScalarMappable(norm=colors.Normalize(0, 180), cmap=color_palette)
    data = np.concatenate([
                            np.array([nm_per_pxl*sparse_peaks_mat.coords[0],
                                      nm_per_pxl*sparse_peaks_mat.coords[1]]),
                            255*np.array(list(zip(*cpal.to_rgba(sparse_peaks_mat.coords[2])))[:-1])
                        ]).T
    
    clusterer.fit(data)
    
    # store fit in data frame
    peaks_df = pd.DataFrame()
    peaks_df['coo_x'] = sparse_peaks_mat.coords[0]
    peaks_df['coo_y'] = sparse_peaks_mat.coords[1]
    peaks_df['theta'] = sparse_peaks_mat.coords[2]
    peaks_df['sin_orientation'] = np.sin(np.radians(peaks_df['theta']))
    peaks_df['complex_theta'] = np.exp(2j*np.deg2rad(peaks_df['theta']))
    peaks_df['cluster_num'] = clusterer.labels_
    peaks_df['confidence'] = clusterer.probabilities_
    peaks_df["coo"] = list(zip(peaks_df.coo_x, peaks_df.coo_y))

    # cluster_num = -1 represents noise
    peaks_df.drop(peaks_df[peaks_df['cluster_num'] == -1].index, inplace = True)
    
    
    theta_range = peaks_df.groupby('cluster_num')['sin_orientation'].max() - peaks_df.groupby('cluster_num')['sin_orientation'].min()
    print('Mean (deg) = ' + str(np.rad2deg(np.arcsin(np.mean(theta_range)))))
    print('Variance (deg) = ' + str(np.rad2deg(np.arcsin(np.var(theta_range)))))

    print('Finished in ' + str(np.round(time.time() - start_time, 2)) + ' seconds.')
    
    return clusterer, peaks_df

def peak_distance(theta1, theta2):
    '''
    decent idea christina! but it doens't work because there is a distribution of chord lengths in a given circle that is nonuniform...
    '''
    x1=np.cos(2*np.deg2rad(theta1))
    y1 = np.sin(2*np.deg2rad(theta1))
    
    x2 = np.cos(2*np.deg2rad(theta2))
    y2 = np.sin(2*np.deg2rad(theta2))
    
    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance

