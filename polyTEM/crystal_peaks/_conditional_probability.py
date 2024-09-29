import numpy as np
import time
import sparse
import multiprocessing as mp
import builtins
import matplotlib.pyplot as plt
from pytem import spatial
from scipy.stats import rv_discrete
import logging

class ConditionalProbability:
    """
    Attributes:
        d: numpy narray of distances, in pixels
        dtheta: numpy array of misorientation angles
        resolution: float, nm/pxl
        cond_prob_mat:2d numpy array with shape (dtheta,d) representing P(dtheta|d)
        pdf: list of scipy.rv_discrete for P_d(dtheta) for each d
    """
    
    def __init__(self, d=None, dtheta=None, pvalues=None, resolution=1):

        self.d = d
        self.dtheta = dtheta
        self.cond_prob_mat = pvalues
        self.resolution = resolution
        self.pdf = None
        return
    
    def plot(self, figsize=[7,7], levels=[0,0.01,0.02,0.05,0.1,0.2,0.5,1], colors='reds',xmax=25,ymax=20):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        fig.set_size_inches(figsize,forward=True)
        logging.debug(f'resolution {self.resolution}')
        logging.debug(f'{self.d}')
        X,Y = np.meshgrid(self.d*self.resolution,self.dtheta)
        Z = self.cond_prob_mat
        
        if colors == 'reds':
            col = ['white','seashell','mistyrose','lightcoral','indianred','firebrick','red']
        elif colors == 'blues':
            col = ['white','azure','lightcyan','paleturquoise','mediumturquoise','lightseagreen','teal']
        else:
            logging.warning('color scheme must be "blues" or "reds". Defaulting to "reds".')
            col = ['white','seashell','mistyrose','lightcoral','indianred','firebrick','red']

        plt.contour(X,Y,Z, colors=col, levels=levels)
        plt.contourf(X,Y,Z, colors=col, levels=levels)
        plt.xlim([0,xmax])
        plt.ylim([0,ymax])
        plt.xlabel('Radial Distance (nm)')
        plt.ylabel('$\Delta\Theta$')
        plt.colorbar()
        plt.show()
        return fig
    
    def discrete_pdf(self):
        '''
        Get Probability Density Function Pd(theta)
        
        This actually calculates the probability mass function since we have discrete integers
        
        Returns:
            pdf_list: list with length (# of distances)
        '''
        pdf_list = []
        # Iterate through the columns which represent the radial distance d
        for col in range(self.cond_prob_mat.shape[1]):
            try:
                pdf_list.append(rv_discrete(name=f'{col} pxls',values=(self.dtheta,self.cond_prob_mat[:,col])))
            except ValueError:
                # occurs when the probabilities don't add up to 1
                # This usuaully happens for the really large distances, so not that it matters much
                logging.debug(f'Col {col} for distance {self.d[col]}')
                pdf_list.append(rv_discrete(name=f'{col} pxls blank'))
        self.pdf = np.array(pdf_list)
        return
    
    def expected_value(self, func):
        '''
        INPUT:
        pdf_list = list of length d containing scipy.stats.rvdiscrete objects for P_d(dtheta)
        --
        OUTPUT:
        ev = 1d array of length d with the expected values of each conditional probability distribution
        '''
        try:
            ev = np.zeros(len(self.pdf))
        except TypeError: # self.pdf is None
            self.discrete_pdf()
            ev = np.zeros(len(self.pdf))
        for i,pdf in enumerate(self.pdf):
            if 'blank' in pdf.name:
                logging.warning('pdf name is blank, probabilities did not add to 1.')
                ev[i] = np.nan
            else:
                try:
                    ev[i] = pdf.expect(func)
                except Exception as e:
                    logging.debug(f'Row {i}')
                    logging.error(traceback.format_exc())

        return ev

#################################################################
#### HELPER FUNCTIONS
#################################################################
def make_distances_array(m, n):
    # adapted from camila's code
    xx, yy = np.abs(np.meshgrid(np.arange(0, m), np.arange(0, n), sparse=False))
    zz = np.round(np.sqrt(xx ** 2 + yy ** 2), 2)

    return zz

def get_unique_dist(shape):
    distances = make_distances_array(shape[0],shape[1])
    return np.unique(distances)

def multiprocessing_count_func(slice, distance_mat, possible_distances, slice_num):
    '''
    this function counts all of the peak pairs represented in slice and sorts them based on
    distance (d) and angle difference (delta_th).  It does so by shifting the coordinates of
    the original slice based on d and delta_th, and then comparing matches.
    --
    INPUTS:
    slice = a 3D sparse matrix with dimensions (max_x,max_y,180) that records locations of each peak
    distance_mat = a matrix that shows the distance from the origin point to each point in the matrix
    possible_distances = a list of all the unique possible distances in the distance matrix
    slice_num = scalar representing which slice of the original peaks matrix is being operated on
    --
    OUTPUT:
    count_mat = a 2D matrix of (d,delta_th) of the number of peak-pairs that satisifies (d,delta_th)
    '''
#     start_time = time.time()
    # this is a slice of the sparse matrix containing all the locations and angles of the peaks
    # coordinates: (x,y,th)
    # the size of this slice is equal to the max_distance that we're interested in (4max_x,4max_y,180)
    # slices need to overlap by max_x
    
    # to find counts, I'm shifting the coords of the sparse matrix over based on delta_d and delta_th
    #
    count_mat = np.zeros(shape=(len(possible_distances), 90))
    coo = builtins.set(zip(*slice.coords))
#     slice.shape = (slice.shape[0] * 2, slice.shape[1]*2, slice.shape[2])
    for delta_th in range(90):
        for d_index, delta_d in enumerate(possible_distances):
            #shift and then compare the coords
            for shift_x, shift_y in np.argwhere(distance_mat == delta_d):
                shifted_slice_coords = (slice.coords[0] + shift_x,
                                        slice.coords[1] + shift_y,
                                        (slice.coords[2] + delta_th) % 180)
                shifted_slice_coords = builtins.set(zip(*shifted_slice_coords))
                count_mat[d_index,delta_th] += len(coo.intersection(shifted_slice_coords))

    return count_mat
            
    
def get_counts_multiprocessing(list_of_slices, threads=8):
    '''
    Uses multiprocessing to run function multiprocessing_count_func, which gets the counts of
    the peak pairs based on distance and delta_theta
    '''
    distance_mat = make_distances_array(list_of_slices[0].shape[0], list_of_slices[0].shape[1])
    unique_distances = np.unique(distance_mat) # Given in Pixels
    start_time = time.time()
    result_list = []
    
    pool = mp.Pool(processes = threads)
    
    for slice_num,slice in enumerate(list_of_slices):
        results = pool.apply_async(multiprocessing_count_func, 
                         args = (slice, distance_mat, unique_distances, slice_num))
        result_list.append(results)
    result_list = [r.get() for r in result_list ]
    pool.close()
    pool.join()
    print('Conditional Probability Matrix Finished in ' + str(np.round(time.time() - start_time,2)) + ' seconds.')
    return(result_list)

def get_cond_prob(sparse_peaks_mat, plot = True, num_slices = 16, resolution = 1, threads=8,**kwargs):
    '''
    Returns a 2d array of (d, dtheta)
    where the d values are incremented based on unique possible distances in the spatial slice
    and the dtheta is incremented by 1 degree.
    the values in the array correspond to Probability of DeltaTheta=dtheta given Distance=d
    
    Args:
        sparse_peaks_mat: 3D sparse array object
        plot: bool
        num_slices: int, number of subimages to divide in (x,y) space, also limits the max distance!
        resolution: float, nm/pxl
        threads: int, number of multiprocessing cores
        kwargs: plot arguments. see ConditionalProbability.plot()
    sparse_meaks_mat= 3D Sparse array object
    plot=B
    '''
    # create list of slices from sparse peaks mat
    # for test methods
#     slices_list = [sparse_peaks_mat for i in range(10)]

    # the goal is two run the 8 cores twice, so divide the sparse matrix into 16 slices
    
    row_size = sparse_peaks_mat.shape[0] / num_slices
    col_size = sparse_peaks_mat.shape[1] / num_slices
    slices_list = []
    for i in range(num_slices):
        for j in range(num_slices):
            slices_list.append(
                sparse_peaks_mat[i*row_size:(i+1)*row_size, j*col_size:(j+1)*col_size, :])
    
    # get counts              
    results = get_counts_multiprocessing(slices_list, threads)
    joint_counts = sum(results)
    distance_counts = np.sum(joint_counts, axis = 1)
    cond_prob_mat = joint_counts / distance_counts[:,np.newaxis]
    cond_prob_mat = cond_prob_mat.T # convert so the shape (M,N) follows (d,dtheta)
    
    # get appropriate distance values for x-axis
    d = np.unique(make_distances_array(slices_list[0].shape[0],slices_list[0].shape[1]))
    # check dimensions
    if d.shape[0] != cond_prob_mat.T.shape[0]:
        logging.error('Conditional Probability Matrix and Distance Array does not match')        
     
    cp = ConditionalProbability(d=d, dtheta=range(90), pvalues=cond_prob_mat, resolution=resolution)
    
    if plot:
        palette=kwargs.get('plot_color','reds')
        max_d=kwargs.get('plot_dist_max',25)
        max_dtheta = kwargs.get('plot_th_max',20) 
        cp.plot(colors=palette,xmax=max_d,ymax=max_dtheta)
        
    return cp



def compare_to_random(cond_prob_mat, reference="uniform", ref_stack=None, plot=False):
    """
    Uses the Kolmogorov-Smirnov test to compare the
    conditional-probability P(Theta | d),
    (where Theta is the difference in orientation,
    and d is the distance in real space, )
    against the uniform distribution.
    
    The Kolmogorov-Smirnov test compares the CDF of
    P(Theta|d) versus the CDF of the uniform
    --
    Output:
    dist_mat = np.ndarray with the same shape as cond_prob_mat
        contains the values of the difference between the CDFs
        at each distance d
    D_list = list of KS-test D, values for each distance d
    """
    dist_mat = np.zeros_like(cond_prob_mat)
    D_list = []
    theta_list = []

    for row in range(cond_prob_mat.shape[0]):
        if reference == "discrete uniform":
            ref_dist = np.cumsum(ref_stack.cond_prob_mat[row,:])
        else:
            ref_dist = None
        dist, D, theta = spatial.stats._ksdist(cond_prob_mat[row,:], reference, ref_dist)
        dist_mat[row,:] = dist
        D_list.append(D)
        theta_list.append(theta)
        
    if plot:
        plt.imshow(dist_mat.T, origin="lower")  
        plt.ylabel('$\Delta \Theta$ (deg)')
        plt.xlabel('Distance (pxl)')
        plt.colorbar()
        plt.clim(0.05,0.5)
        plt.title(r'KS dist CDF(sample) - CDF(uniform)')
        plt.show()
        
    return dist_mat, D_list, theta_list



