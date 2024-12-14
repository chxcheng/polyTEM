from .crystal_stack import CrystalStack
from . import vector_graph_utils
from .. import utilities
import numpy as np
from scipy.stats import sem
import logging
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
from tqdm.autonotebook import tqdm
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import sparse
from functools import partial
import itertools
import networkx as nx

## Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008


class CrystalStack3D():
    """
    Assumptions:
    * the sample datacube parameters were used to generate each of the crystal stacks. Thus, the resolution and shape is the same.
    
    Attributes:
        stack_list: list of crystal stacks
        tilt_angles: list of  tilt angles corresponding to each stack in stack_list
        domain_collections: list of DomainCollections correspond to each stack in stack_list
        shape: (x,y) shape 
        resolution: resolution of tilt stack (afer sliding fft)
        vector_field: each row is from one fft peak and contains (x,y,a,b,c) for vector field direction
        threshold_func: function for peak finding
        Kb, Ks: Frank Elastic Constants in Angstroms
    
    """
    def __init__(self,stack_list,tilt_angles,domain_collections,auto_populate=True,orient_perpendicular=False):
        self.stack_list = stack_list
        self.tilt_angles = tilt_angles
        self.domain_collections = domain_collections
        self.shape = self.stack_list[0].sparse_peaks_mat.shape[0:2]
        self.resolution = stack_list[0].resolution
        self.is_aligned = False
        
        if auto_populate:
            self.create_vector_field(orient_perpendicular)
            self.estimate_bending_stiffness(orient_perpendicular)
        return
           
    def save(self,savedir,prefix=''):
        """
        Saves the stacks based on their individual outdirs, with the prefix inserted in front of crystalstack.pkl
        Saves the tilt angles at savedir, tilt_angles.json
        Saves the vector field at savedir, vector_field.json
        """
        for stack in tqdm(self.stack_list):
            stack.save(prefix=prefix)
        jsonpickle_numpy.register_handlers()
        utilities.saveas_json(self.tilt_angles,os.path.join(savedir,f'{prefix}tilt_angles.json'))
        utilities.saveas_json(self.vector_field,os.path.join(savedir,f'{prefix}vector_field.json'))
        return    
    
    def create_vector_field(self,rotate90_orientation=False):
        """
        Creates vector field (x,y,alpha,a,b,c), 
        where x is the x-axis of the image (across the columns of the data)
              y is the y-axis of the image (across the rows of the data)
              alpha is the tilt angle
              a,b,c represents the unit vector field of chain orientation, centered at (x,y)
        """
        vector_field_list = []
        for stack,alpha in zip(self.stack_list,self.tilt_angles):
            alpha_rad = np.deg2rad(alpha)
            thetas = stack.sparse_peaks_mat.coords[2,:]
            thetas_rad = np.deg2rad(thetas)
            if rotate90_orientation:
                a = -np.sin(thetas_rad)*np.cos(alpha_rad)
                b = np.cos(thetas_rad)
                c = -np.sin(thetas_rad)*np.sin(alpha_rad)
            else:
                b = np.sin(thetas_rad)
                a = np.cos(thetas_rad)*np.cos(alpha_rad)
                c = np.cos(thetas_rad)*np.sin(alpha_rad)
            vector_field_list.append(np.stack((stack.sparse_peaks_mat.coords[1,:],
                                     stack.sparse_peaks_mat.coords[0,:],
                                     np.repeat(alpha,len(thetas)),
                                     a,b,c)).T
                                    )
        vector_field = np.concatenate(vector_field_list,axis=0)
        self.vector_field = vector_field
        self.vector_field_list = vector_field_list
        return vector_field
    
    def align_vector_field(self,window_size,step_size,alignment_result,alignment_indices,origin_shift = None):
        """
        Shifts the (x,y) coordinates in the vector field based on the marker alignment results
        and convert to (u,v) where u = x/cos(alpha) and v=y
        
        this returns a vector field where (u,v) has the pixel resolution of the original image, 
        before downsampling with the sliding fft
        
        Args:
            window_size: int, parameter used during sliding fft scan
            step_size: int, parameter used during sliding fft scan
            alignment_result: AlignmentResult object
            alignment_indices: the image index in AlignmentResult that corresponds to the images used to create the CrystalStack3D
            origin_shift: any origin shift applied during alignment
            
        Returns:
            vector field containing (u,v,alpha,a,b,c) of each fft peak feature
        """
        if origin_shift:
            if len(origin_shift) == 1:
                x_origin_shift = origin_shift
                y_origin_shift = origin_shift
            else:
                x_origin_shift = origin_shift[0]
                y_origin_shift = origin_shift[1]
        else:
            x_origin_shift = 0
            y_origin_shift = 0
            
        # remap the x and y in the down sampled vector field back to the center of the fft window
        # then the x and y values in the vector field will match the correct index of the window_centers array
        window_centers_x  = (window_size/2) + step_size*np.arange(self.shape[1])
        window_centers_y = (window_size/2) + step_size*np.arange(self.shape[0])
        
        # map the AlignmentResult indices with the images used in the tilt stack
        alignment_indices = find_match(x=np.array(alignment_result.tilt_angles), 
                                       y=np.array(self.tilt_angles))
            
        for vf_index,image_index in enumerate(alignment_indices):
            vector_field = self.vector_field_list[vf_index] 
            x_indices = vector_field[:,0]
            y_indices = vector_field[:,1]
            shifted_x = window_centers_x[x_indices.astype(int)] - x_origin_shift
            shifted_y = window_centers_y[y_indices.astype(int)] - y_origin_shift
            shifted_coo = np.stack((shifted_x,shifted_y,np.ones((len(shifted_x)))),axis=1) #format for openCV warpAffine
            
            # shift based on the alignment using affine transform
            M = alignment_result.affine_transform_mat[image_index,:,:]
            aligned_coo = shifted_coo.dot(M.T)
            
            # convert x to u
            alpha = vector_field[:,2]
            u = aligned_coo[:,0]/np.cos(np.deg2rad(alpha)) 
            
            vector_field[:,0] = u
            vector_field[:,1] = aligned_coo[:,1]
            
        self.vector_field = np.concatenate(self.vector_field_list,axis=0)
        self.is_aligned = True

        return self.vector_field
    
    def find_peak_threshold(self, alpha, inverse = False, ):
        """
        Use the fft intensities for high tilt angles to estimate noise floor to set threshold function
        for fft peak finding of lower tilt angles
        
        Args:
            alpha: the angle cutoff to divide the tilt angles
            inverse: default False, selects tilt angles greater than alpha. if True, selects tilt angles smaller.
        
        Returns:
            threshold_func: lambda function 
        """
        if inverse:
            high_angle_indices = np.where(np.abs(self.tilt_angles) < alpha)[0]
        else:
            high_angle_indices = np.where(np.abs(self.tilt_angles) > alpha)[0]
        mean_list = []
        std_list= []
        for i,index in enumerate(high_angle_indices):
            fft_ii = self.stack_list[index].datacube
            mean_list.append(np.nanmean(fft_ii))
            std_list.append(np.nanstd(fft_ii))
        logging.debug(f'Average mean is {np.mean(mean_list):2f} and average standard deviation is {np.mean(std_list):2f}')

        self.threshold_func = lambda x: np.mean(mean_list) + 3*(np.mean(std_list))
        return self.threshold_func
    
    def apply_threshold_func(self, threshold_func = None):
        """
        Sets the threshold function of every stack in the stack list
        
        Args:
            threshold_func: default None, uses self.threshold_func, otherwise input argument will override
        """
        if not threshold_func:
            threshold_func = self.threshold_func
        for stack in self.stack_list:
            stack._threshold_function = threshold_func
        return
    
    def estimate_bending_stiffness(self, flip=False):
        """
        Estimates the bending stiffness by calculating the orientation correlation length
        Under the Worm-Like-Chain model, the orientation correlation length == bending stiffness
        
        Args:
            flip: bool, swaps perpendicular and parallel orientations. i.e. set True if
                    director orientation was obtained as "lamellar" but was actually "backbone"
        Returns:
            Ld: average orientation correlation length
            stderr: standard error of the mean of Ld
            
        """
        ## get Ld_par and Ld_perp as a function of tilt angle alpha
        low_angle_domains = list(itertools.compress(self.domain_collections,
                                                    list(np.abs(np.array(self.tilt_angles))<30)))
        Ld_par_nm = np.array([domain.Ld_par for domain in low_angle_domains],dtype=float)
        Ld_par_nm *= self.resolution
        Ld_perp_nm = np.array([domain.Ld_perp for domain in low_angle_domains],dtype=float)
        Ld_perp_nm *= self.resolution
        
        if flip:
            self.Kb = Ld_perp_nm * 10
            self.Ks = Ld_par_nm * 10
        else:
            self.Kb = Ld_par_nm * 10
            self.Ks = Ld_perp_nm * 10
        return
               
    
    def adjacency_probability_matrix(self,savedir,resolution=1.15,max_distance_pxl=32,alpha_cutoff=70, threads=8, node_group_size=100,model="nematic",**kwargs):
        """
        saves the adjacency probabilities in the directory 'savedir' with the format
        {savedir}/node{##}.npz, where ## is the starting row index for that submatrix.
        Args:
            savedir (str): Directory to save adjacency probability sub-matrices
            Resolution: original imaging resolution of raw TEM image
            max_distance: int, pixel limit to build possible connections
            alpha_cutoff: int, tilt angle cutoff to build possible connections
            threads: int, for multiprocessing
            node_group_size: int, to segment nodes for multiprocessing
            model: "single-chain" or "nematic"
            **kwargs: keyword-argument parameters for bending probability function
            --
            Lp: float, A, persistence length for "single-chain" model
            Kb, Ks: Frank Elastic Constants for nematic, dimensionless

        """
        max_distance = max_distance_pxl*resolution
        n_total = self.vector_field.shape[0]

        start_time = time.time()   
        result_list = []
        pool = mp.Pool(processes=threads)
        error_handler = partial(_terminate_pool_on_error, pool)
        
        for start_index in np.arange(n_total,step=node_group_size):
            if start_index + node_group_size > n_total:
                end_index = -1
            else:
                end_index = start_index + node_group_size
            node_group = self.vector_field[start_index:start_index+node_group_size,:]
            results = pool.apply_async(_multiprocessing_adjacency_func,
                                       (node_group, start_index, self.vector_field,
                                        max_distance, alpha_cutoff, n_total, savedir),
                                       kwargs,
                                       error_callback=error_handler,
                                      )
            result_list.append(results)

        result_list = [r.get() for r in result_list ]
        pool.close()
        pool.join()
        print('Adjacency Probability Matrix Finished in ' + str(np.round(time.time() - start_time,2)) + ' seconds.')
        
        return
        
    def build_vector_graph(self, file_dir, node_range):
        """
        Build vector graph representation from adjacency probability matrix.
        
        The Adjacency Probability Matrix is stored in the format {prob_dir}/node{##}.npz
        where the ## is the start index of the first node in that numpy array.
        The graph is built by iterating through the probabilities in
        {prob_dir}/node{start_index) for start_index in node_range
        
        Args:
            file_dir (str): Directory
            node_range (range): the set of starting indexes for all of the .npz files to iterate through
                e.g. np.arange(0,total_nodes,node_group_size) 
                where node_group_size was set in CrystalStack3D.adjacency_probability_matrix()
        
        Returns:
            networkx graph
        """
        if not self.is_aligned:
            logging.error('Cannot build Vector Graph on unaligned vector field!')
            return
        else:
            graph = vector_graph_utils.create_nodes_from_vectorfield(self.vector_field)
            for start_index in tqdm(node_range):
                probs_mat = sparse.load_npz(os.path.join(file_dir,f'node{start_index}.npz'))
                vector_graph_utils.create_edges_from_simulation(graph, probs_mat, start_index)
            return graph
        
        
##########################################
## FUNCTIONS
##########################################

def find_match(x,y):
    """
    For each element in y, get the index of x that matches that element
    Used to match the tilt_angles between the images used to create the CrystalStack3D and those used in the AlignmentResult
    
    Usage: x should be AlignmentResult tilt angles and y should be CrystalStack3D tilt angles
    
    Args:
        x,y: numpy array
    Returns:
        indices: numpy array
    """
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    indices = np.take(index, sorted_index, mode="clip")
    return indices

def _terminate_pool_on_error(pool, err):
    print(f"ERROR: {err}")
    pool.terminate()
    return

def _multiprocessing_adjacency_func(node_group, start_index, vector_field, max_distance,alpha_cutoff, n_total, savedir, model="nematic", **kwargs):
    """
    multiprocessing function called to create adjacency probability matrix using Pool.apply_async()
    builds the upper triangle of the symmetric adjacency probability matrix, so vector_field 
    
    Because this is intended to use with apply_async() and order is lost,
    A tuple is returned that includes to node index so that all of the resulting adjacency_p_rows
    can be sorted and reordered properly
    
    Args:
        node_group: np.ndarray a group of rows from CrystalStack3D.vector_field
        start_index: int, node_index of first node in node_group
        vector_field: np.array of tilt_stack.vector_field 
        max_distance: float A, cutoff for connections
        alpha_cutoff: float, tilt angle cutoff for connections
        n_total: number of nodes
        savedir = str, where to save the probability adjacency matrix for this node

        **kwargs
        --
        model: "single-chain" or "nematic" to determine bending probability. Default "nematic"
        Lp: Float, persistence length in A for "single-chain" bending probability
        Kb, Ks: float, Frank Elastic Constants for "nematic" bending probability
        
    
    Returns:
        start_index: index of node
        adjacency_p_row: sparse array with shape(n_total) with adjacency probabilities 
    """
    from .. import spatial as pyspace
    full_p_mat = np.zeros(shape=(node_group.shape[0],n_total))
    for i,node in enumerate(node_group):
        node_index = start_index + i
        t0 = node[3:6]
        x0 = node[0]
        y0 = node[1]
        alpha0=node[2]

        # identify connected nodes
        index_set_x = (np.abs(vector_field[:,0]-x0) < max_distance)
        index_set_y = (np.abs(vector_field[:,1]-y0) < max_distance)
        index_set_alpha = (np.abs(vector_field[:,2]-alpha0) <= alpha_cutoff)
        connected_indices= np.nonzero(np.logical_and(index_set_x,index_set_y,index_set_alpha))[0]

        for j_index in connected_indices[connected_indices>node_index]:
            node2 = vector_field[j_index,:]
            full_p_mat[i,j_index] = pyspace.stats.bending_probability(node,node2,model=model,**kwargs)
        
    sparse_p_mat = sparse.COO.from_numpy(full_p_mat)
    sparse.save_npz(os.path.join(savedir,f'node{start_index}.npz'),sparse_p_mat)

    return start_index, sparse_p_mat