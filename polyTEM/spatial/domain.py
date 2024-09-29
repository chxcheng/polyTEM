"""
To define a cluster object

TO DO:
- move spatial_analysis into spatial module
- change create_polygons and cluster_df output to create Domain objects
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors  as mcolors

from scipy.interpolate import CloughTocher2DInterpolator
from scipy import ndimage
from scipy.stats import rv_histogram
from scipy.integrate import IntegrationWarning
from scipy.fft import dct

from . import stats
from .. import utilities

import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import jsonpickle.ext.pandas as jsonpickle_pandas
jsonpickle_pandas.register_handlers()
import copy

import alphashape #using v 1.3
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiPoint # Using v1.8, do not switch to v2.0+, not compatible with alphashape 1.3
import multiprocessing as mp

import warnings
from shapely.errors import ShapelyDeprecationWarning
import os
import logging
from tqdm.autonotebook import tqdm


class DomainCollection():
    """
    A collection of one or more Domains
    
    Attributes:
        domains: list of Domain objects
        savedir: path to save directory
        pixel_resolution: nm/pixel
        parallel_dtheta: list of scipy random variable class generated from rv_histogram, 
                which contains the distribution for probability P(dtheta|r), where r is the list index
                (i.e. parallel_dtheta[1] = P(dtheta | r=1) in the direction parallel to the nematic director of the centroid)
        perpendicular_dtheta: same as DomainCollection.parallel_dtheta but in the perpendicular direction
        Ld_par: orientation correlation length in the parallel direction (in pixels)
        Ld_perp: orientation correlation length in the perpendicular direction (in pixels)
        
    Methods:
        histogram_dtheta: computes parallel_dtheta and perpendicular_dtheta
        plot_orientation_change: plots the pdf from histogram_dtheta
        
    Private attributes:
        _parallel_dtheta_values, _perpendicular_dtheta_values
        _expected_val_par, _expected_val_perp
        _model_Ld_par, _model_Ld_perp
        _orientation_correlation_length_bounds
    """
    ## From test, 1344 domains in list has 11kB
    def __init__(self,domains:list=None,pixel_res=None,savedir=''):
        self.domains=domains
        self.savedir=savedir
        self.pixel_resolution=pixel_res
        
    @classmethod
    def from_crystalstack(cls,crystal_stack,savedir='',alpha=0.2,cluster_df=None, num_threads=1):
        """
        Creates DomainsCollection from CrystalStack
        
        for 1400 domains, this takes 12 seconds on 1 core if no cluster_df is given.
        Takes 6 seconds if cluster_df is provided (Polygons pre computed)
        So I probably don't need to speed it up.
        but the multiprocessing is written for num_threads>1 just in case.
        
        Arguments:
            crystal_stack: CrystalStack object
            savedir: directory for savefiles
            alpha: parameter to extract shape of domain, see Domain.make_domain()
            cluster_df: optional, backwards compatibility with cluster_dfs made by pytem.spatial_analysis.create_polygons()
            num_threads: integer, num_threads == 1 core does not use multiprocessing, else sets the number of cores used
        """
        if num_threads > 1:
            # use multiprocessing.Pool to speed up creation of Domains
            pool = mp.Pool(processes=num_threads)
            pool_results = []
            for cluster_num in sorted(crystal_stack.peaks_df['cluster_num'].unique()):
                if cluster_df is not None:
                    polygon = cluster_df.query(f'cluster_num=={cluster_num}')['geometry'].item()
                else:
                    polygon = None
                pool_results.append(
                    pool.apply_async(
                        Domain.make_domain,
                        args = (crystal_stack,cluster_num,savedir,alpha,polygon) ) )
            pool_results = [r.get() for r in pool_results]
            pool.close()
            pool.join()
        else:
            pool_results = []
            for cluster_num in tqdm(crystal_stack.peaks_df['cluster_num'].unique()):
                if cluster_df is not None:
                    polygon = cluster_df.query(f'cluster_num=={cluster_num}')['geometry'].item()
                else:
                    polygon = None
                pool_results.append(
                    Domain.make_domain(crystal_stack,cluster_num,savedir,alpha,polygon) )                   
            
        domains_collection = cls(domains = pool_results,
                                 pixel_res = crystal_stack.resolution,
                                 savedir = savedir)
        return domains_collection
                          
    @classmethod
    def load(cls,savefile):
        domains = utilities.load_json(savefile)
        
        ## recreate rv_histograms for parallel_dtheta and perpendicular_dtheta
        domains.parallel_dtheta = stats.make_histogram_distributions(domains._parallel_dtheta_values, bins=np.arange(-90,90,1))
        domains.perpendicular_dtheta = stats.make_histogram_distributions(domains._perpendicular_dtheta_values, bins=np.arange(-90,90,1))
        
        return domains
    
    def save(self,savefile):
        """
        Save as json. However, scipy.stats.distributions cannot be serialized, 
        so DomainCollection.parallel_dtheta and DomainCollection.perpendicular_dtheta cannot be saved.
        Instead, we will only save _parallel_dtheta_values and _perpendicular_dtheta_values, 
        and the rv_histogram will be recreated upon loading
        """
        jsonpickle_numpy.register_handlers()
        jsonpickle_pandas.register_handlers()
        
        ## remove unserializable attributes
        serializable_item = self._remove_unpicklable()

        if savefile:
            utilities.saveas_json(serializable_item,savefile)
        else:
            utilities.saveas_json(serializable_item,os.path.join(savedir,'domainsCollection.json'))
        return
    
    def _remove_unpicklable(self):
        """
        Returns a copy of the DomainCollection, but with unpicklable attributes set to None
        
        Current List of unserializable attributes:
            parallel_dtheta and perpendicular_dtheta: scipy.stats distributions cannot be serialized
        """
        collection_to_save = copy.deepcopy(self)
        collection_to_save.parallel_dtheta=None
        collection_to_save.perpendicular_dtheta=None
        return collection_to_save
        
    
    def histogram_dtheta(self):
        """
        Histograms the orientation change as a function of distance in the directions
        parallel and perpendicular to the nematic director of at the domain centroid
        """
        ## Initialize the arrays containing the delta thetas
        num_domains = len(self.domains)
        max_dimensions = np.max([domain.aligned_data.shape for domain in self.domains],
                                axis=0)
        max_num_distances_par = max_dimensions[1]//2
        par_dtheta_array = np.full(shape=(2*num_domains,max_num_distances_par),
                                 fill_value=np.nan)
        max_num_distances_perp = max_dimensions[0]//2
        perp_dtheta_array = np.full(shape=(2*num_domains,max_num_distances_perp),
                                    fill_value=np.nan)
        
        # Get delta theta values as a function of distance
        for domain_index,domain in enumerate(self.domains):
            array = domain.aligned_data
            centerx = (array.shape[0]-1)//2
            centery = (array.shape[1]-1)//2
            ## Get DeltaTheta as a function of distance
            for distance in range(centery): #in par direction
                par_dtheta_array[2*domain_index,distance] = array[centerx,centery+distance]
                par_dtheta_array[2*domain_index + 1,distance] = array[centerx,centery-distance]
            for distance in range(centerx): #in perp direction
                perp_dtheta_array[2*domain_index,distance] = array[centerx+distance,centery]
                perp_dtheta_array[2*domain_index + 1,distance] = array[centerx-distance,centery]
                
        self._parallel_dtheta_values = par_dtheta_array
        self._perpendicular_dtheta_values = perp_dtheta_array
        
        # histogram the delta theta values and convert into rv_histogram distributions
        self.parallel_dtheta = stats.make_histogram_distributions(self._parallel_dtheta_values, bins=np.arange(-90,90,1))
        self.perpendicular_dtheta = stats.make_histogram_distributions(self._perpendicular_dtheta_values, bins=np.arange(-90,90,1))
        return
                
    def plot_orientation_change(self, distances:list, dtheta_min=-5, dtheta_max=5):
        """
        Plots the pdf of the probability $P(\Delta\theta | r)$ in the directions parallel and perpendicular
        to the nematic director of the domain centroid.
        
        Arguments:
            distances: list of integer distances to plot
            dtheta_min, dtheta_max: int
        """
        x = np.linspace(dtheta_min,dtheta_max)
        plt.figure(figsize=(10,5))
        for distance in distances:
            plt.subplot(1,2,1)
            plt.plot(x, self.parallel_dtheta[distance].pdf(x), label=distance)

            plt.subplot(1,2,2)
            plt.plot(x,self.perpendicular_dtheta[distance].pdf(x), label=distance)

        plt.subplot(1,2,1)
        plt.xlabel(r'$\Delta\theta$')
        plt.title(r'$P_\parallel(\Delta\theta | r)$')
        plt.subplot(1,2,2)
        plt.xlabel(r'$\Delta\theta$')
        plt.title(r'$P_\perp(\Delta\theta | r )$')
        plt.legend()
        return
    
    def orientation_correlation_length(self,min_dist=0,max_dist=20,plot=False):
        """
        Calculates the orientation correlation length given by the exponential decay fit of
        the expected value <2cos^2(\Delta\theta)-1>
        """        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            self._orientation_correlation_length_bounds = [min_dist,max_dist]
            self._expected_val_par = np.array([dist.expect(stats.legendre_polynomial_2(dimension=2),limit=100)
                                         for dist in tqdm(self.parallel_dtheta)])
            self._expected_val_perp = np.array([dist.expect(stats.legendre_polynomial_2(dimension=2),limit=100)
                                          for dist in tqdm(self.perpendicular_dtheta)])

            self.Ld_par, self._model_Ld_par = stats.exponential_decay_length(y=self._expected_val_par,
                                                    x=np.arange(len(self.parallel_dtheta)),
                                                    start_ind=min_dist,
                                                    end_ind=max_dist)
            self.Ld_perp, self ._model_Ld_perp = stats.exponential_decay_length(y=self._expected_val_perp,
                                                    x=np.arange(len(self.perpendicular_dtheta)),
                                                    start_ind=min_dist,
                                                    end_ind=max_dist)
        if plot:
            self.plot_orientation_correlation()
        return
        
    def plot_orientation_correlation(self):
        lb,ub = self._orientation_correlation_length_bounds
        fig = plt.figure(figsize=(10,4))
        plt.suptitle('Orientation Correlation')
        
        plt.subplot(1,2,1)
        plt.title('Parallel Direction')
        x=np.arange(len(self.parallel_dtheta))
        plt.plot(x,np.log(np.abs(self._expected_val_par)),
                 'c-',alpha=0.5, linewidth=1)
        plt.plot(x[lb:ub],self._model_Ld_par.intercept + self._model_Ld_par.slope*x[lb:ub],
                 'r--',alpha=1, linewidth=1)                   
        plt.xlabel('Distance (pxl)')
        plt.ylabel(r'$\ln(\langle 2 \cos^2(\Delta\theta) - 1 \rangle)$')

        plt.subplot(1,2,2)
        plt.title('Perpendicular Direciton')
        x=np.arange(len(self.perpendicular_dtheta))
        plt.plot(x, np.log(np.abs(self._expected_val_perp)),
                 'c-',alpha=0.5, linewidth=1)
        plt.plot(x[lb:ub], self._model_Ld_perp.intercept + self._model_Ld_perp.slope*x[lb:ub],
                 'r--',alpha=1,linewidth=1)
        return fig

    def frank_elastic_relation(self,freqs=np.linspace(0.14,1)):
        """
        Calculate an estimate for the Frank elastic relations by taking the average
        of the Fourier Transform of the orientation correlation function
        
        Let R = autocorrelation function = $\langle n(r) \cdot n(r')\rangle$. <br>
        Then the Fourier transform of R is: 
        \begin{equation}
        \mathscr{F}\{R\}(k) = 1 - \frac{1}{K_B k_y^2 + K_Sk_x^2}
        \end{equation}

        If we use R_parallel and R_perpendicular instead we get:
        \begin{align}
            \mathscr{F}\{R_{\parallel}\}(k) = 1 - \frac{1}{K_B k_y^2}\\
            \mathscr{F}\{R_{\perp}\}(k) = 1 - \frac{1}{K_S k_x^2}
        \end{align}
        and the Frank Elastic relations 
        \begin{align}
            K_Bk_y^2 &= \frac{1}{1-\mathscr{F}\{R_{\parallel}\}}\\
            K_Sk_x^2 &= \frac{1}{1-\mathscr{F}\{R_{\perp}\}}
        \end{align}
        
        Returns:
            freqs
            Ks_kx^2
            Kb_ky^2
        """
        self.k_freqs = freqs
        
        # remove nan from the end of the array (this was at distances without enought data for expected val)
        R_par = self._expected_val_par[~np.isnan(self._expected_val_par)]
        FR_par = dct(R_par) 
        freqs_par = np.arange(len(R_par))/len(R_par)
        Kbend_ky2 = 1/(1-FR_par)
        self.Kbend_ky2 = np.interp(freqs,freqs_par,Kbend_ky2)

        R_perp = self._expected_val_perp[~np.isnan(self._expected_val_perp)]
        FR_perp = dct(R_perp)
        freqs_perp = np.arange(len(R_perp))/len(R_perp)
        Ksplay_kx2 = 1/(1-FR_perp)
        self.Ksplay_kx2 = np.interp(freqs,freqs_perp,Ksplay_kx2)

        return self.k_freqs, self.Ksplay_kx2, self.Kbend_ky2
        
                               

    
class Domain:
    """
    Attributes:
        data: np.array of the theta values clustered into this domain
        label: int index label that corresponds to the cluster_num in CrystalStack and cluster_df
        savedir: directory to save to
        polygon: Shapely Polygon object of the domain
        centroid: indices of center of mass, calculated from the Shapely Polygon
        coo: coordinates of the domain
        centroid_orientation: theta orientation at the centroid of the domain
        interpolated_data: np.array of theta values with missing values interpolated, and shifted such that the interpolated centroid is at the image center
        aligned_data: np.array of interpolated_data rotated such that the nematic director of the centroid is in the y-direction
        
    Notes:
        * This class is indexed by x,y == i,j
    """
    def __init__(self, data=None, pixel_res=None,label=None, savedir = '', polygon=None, centroid=None, centroid_orientation=None, coo=None, autopopulate=False):
        self.data = data
        self.pixel_resolution = pixel_res
        self.label = label
        self.savedir = savedir
        self.polygon=polygon
        self.centroid=centroid
        self.centroid_orientation=centroid_orientation
        self.coo=coo
        
        
        ## Autocalculate the following attributes upon class creation
        if autopopulate:
            self.interpolate()
            self.align_nematic_center()
            self.gradient()
            
        return  
    
    def save(self,filename=''):
        jsonpickle_numpy.register_handlers()
        jsonpickle_pandas.register_handlers()
        if not filename:
            filename = os.path.join(self.savedir,f'domain_{self.label}.json')
        utilities.saveas_json(self,filename)
        return
    
    @classmethod
    def load(cls,filename):
        domain = utilities.load_json(filename)
        return domain
    
    @classmethod
    def make_domain(cls,crystal_stack,label,savedir,alpha=0.2,polygon=None):
        """
        Arguments:
            stack: CrystalStack
            label: cluster_num of interest
            savedir: directory to save Domain information
            alpha: default 0.2, hyperparameter for getting domain shape from clustered coordinates, alpha=0 is the Convex Hull, alpha>1 Convex Hull
            polygon: optional, Shapely Polygon object
        Returns:
            Domain object
        """
        cluster = crystal_stack.peaks_df.query(f'cluster_num=={label}')
    
        if polygon is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                try:
                    polygon = alphashape.alphashape(list(cluster['coo']),alpha).buffer(0.2)
                except Exception as e:
                    logging.error(f'Error creating polygon for cluster #{label}')
                    logging.error(f'{e}')
                    return
            
        ## Get the nematic orientation at the center of the domain
        centroid_coo = np.array(polygon.centroid.coords[0])
        distance_from_center = cluster['coo'].apply(lambda x: np.linalg.norm(np.array(x)-centroid_coo))
        center_orientation = cluster.loc[distance_from_center.idxmin(),'theta']

        ## create nematic orientation difference array
        xmin, ymin = np.min([*cluster['coo']],axis=0)
        xmax, ymax = np.max([*cluster['coo']],axis=0)
        orientation_array = np.full((xmax-xmin+1,ymax-ymin+1),np.nan)
        for index,(i,j) in enumerate(cluster['coo']):
            j -= ymin
            i -= xmin
            orientation_array[i,j] = cluster['theta'].to_numpy()[index]

        ## map cluster centroid to indices for orientation_change array
        logging.debug(f'Polygon centroid is {centroid_coo}, with bounding box xmin={xmin}, ymin={ymin}')
        center_indices = np.round(centroid_coo) - [xmin,ymin]
        
        domain = cls(data=orientation_array,
                     pixel_res=crystal_stack.resolution,
                     label=label,
                     savedir=savedir,
                     polygon=polygon,
                     centroid=center_indices.astype(int),
                     centroid_orientation=center_orientation,
                     coo=np.array([*cluster['coo']]),
                     autopopulate=True)
                     
        return domain
    
    def interpolate(self, plot=False):
        """
        Interpolates missing values in the domain array
        """
        input_indices = np.argwhere(~np.isnan(self.data))
        input_values = self.data[~np.isnan(self.data)].flatten()
        cmin = np.min(input_values)
        cmax = np.max(input_values)
        interp = CloughTocher2DInterpolator(input_indices,input_values)
        X = np.arange(0,self.data.shape[0])
        Y = np.arange(0,self.data.shape[1])
        X, Y = np.meshgrid(X, Y, indexing='ij')  # 2D grid for interpolation
        self.interpolated_data = interp(X,Y)
        
        # Pad the array such that the centroid is also at the center of the array
        data_centerx=(self.interpolated_data.shape[0]-1)//2
        data_centery=(self.interpolated_data.shape[1]-1)//2
        if self.centroid[0] != data_centerx or self.centroid[1] != data_centery: 
            ## align centroid and array center
            pad_bottom = 2*max(0,self.centroid[0]-data_centerx)
            pad_top = 2*max(0,data_centerx-self.centroid[0])
            pad_right = 2*max(0,self.centroid[1]-data_centery)
            pad_left = 2*max(0,data_centery-self.centroid[1])
            logging.debug(f'Pad_width = {pad_top,pad_bottom,pad_left,pad_right}')
            self.interpolated_data = np.pad(self.interpolated_data,((pad_top,pad_bottom),(pad_left,pad_right)),'constant',constant_values=np.nan)
        
        if plot:
            plt.figure()
            plt.imshow(interpolated_array,vmin=cmin,vmax=cmax)
            plt.colorbar()
            plt.title('Interpolated')
        return
    
    def rotate(self, angle):
        """
        Rotates the interpolated domain (and the orientations) by specified angle,
        Rotates about the center of the image
        """
        #ndimage.rotate rotates the image about the center of the image.              
        rotated_array = ndimage.rotate(self.interpolated_data,angle,reshape=True,order=0,cval=np.nan) ## rotated domain shape
        rotated_array += angle ## adjust values to correspond to the rotation
        return rotated_array

    def align_nematic_center(self, plot=False):
        """
        rotates the interpolated domain (and the orientations) such that the nematic director
        at the centroid of the domain is aligned parallel to the horizontal axis (j direction)
        """
        aligned_array = self.rotate(-self.centroid_orientation)
        if plot:
            _plot_domain(aligned_array,title='Aligned')
        self.aligned_data = aligned_array
        return
    
    def gradient(self, plot=False):
        """
        Calculates the spatial gradient of the nematic directors in the domain
        This function uses ij indexing, such that x iterates through rows and y iterates through columns and
        the array is plotted using plt.imshow
        """
        extrapolated_array = stats.extrapolate_NN(self.aligned_data)
        
        # change theta into cartesian coordinates
        nx = np.sin(np.deg2rad(extrapolated_array))
        ny = np.cos(np.deg2rad(extrapolated_array))
        n = np.array([nx,ny])
        x = np.arange(0,extrapolated_array.shape[0])
        y = np.arange(0,extrapolated_array.shape[1])
        grad = np.gradient(n,x,y,axis=(1,2))
        # Make it an array and flip first axes such that
        # the first axes indexes the components of the vector n,
        # the second axes indexes the cartesian direction of the gradient
        #     dnx_dy = D[0,1,:,:]
        #     dny_dx = D[1,0,:,:]
        grad = np.array(grad).transpose((1,0,2,3)) 
        
        # remove extrapolated values by applying mask
        masked_grad = np.ma.masked_where(np.tile(np.isnan(self.aligned_data),(2,2,1,1)), grad)

        self.gradient = masked_grad
        
        if plot:
            self.plot_gradient('dx/dx','dx/dy','dy/dx','dy/dy')
            
        return
    
    def plot_gradient(self,*args):
        """
        Arguments:
            *args: which gradients to plot, defined as strings. 
                can be 'dx/dx','dx/dy','dy/dx','dy/dy'
        """
        def match_argument(arg_value):
            if arg_value =='dx/dx':
                return self.gradient[0,0,:,:]
            elif arg_value == 'dx/dy':
                return self.gradient[0,1,:,:]
            elif arg_value == 'dy/dx':
                return self.gradient[1,0,:,:]
            elif arg_value == 'dy/dy':
                return self.gradient[1,1,:,:]
            else:
                raise ValueError("Arguments to plot gradient must be one of: 'dx/dx','dx/dy','dy/dx','dy/dy'")
        
        if len(args) <= 2:
            nrows = 1
        elif len(args) <= 4:
            nrows = 2
        else:
            logging.error("domain.plot_gradient: too many arguments provided")
            return
        ncols=2
        
        plt.figure()
        for index,value in enumerate(args):
            plt.subplot(nrows,ncols,index+1)
            plt.imshow(match_argument(value),norm=mcolors.TwoSlopeNorm(vcenter=0), cmap='RdBu')
            plt.colorbar()
            plt.title(value)
        return
    


#######################
## HELPER FUNCTIONS
#######################

def _plot_domain(array,vmin=None,vmax=None,cmap='RdBu',title=''):
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    plt.figure()
    plt.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    return


    

            
            
