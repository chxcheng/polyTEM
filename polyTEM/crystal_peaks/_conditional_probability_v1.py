import numpy as np
import time
import sparse
import multiprocessing as mp
import builtins
import matplotlib.pyplot as plt
from pytem import statistics as stats
from scipy.stats import rv_discrete
import logging

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
            
    
def get_counts_multiprocessing(list_of_slices):
    '''
    Uses multiprocessing to run function multiprocessing_count_func, which gets the counts of
    the peak pairs based on distance and delta_theta
    '''
    distance_mat = make_distances_array(list_of_slices[0].shape[0], list_of_slices[0].shape[1])
    unique_distances = np.unique(distance_mat) # Given in Pixels
    start_time = time.time()
    result_list = []
    
    pool = mp.Pool(processes = 8)
    
    for slice_num,slice in enumerate(list_of_slices):
        results = pool.apply_async(multiprocessing_count_func, 
                         args = (slice, distance_mat, unique_distances, slice_num))
        result_list.append(results)
    result_list = [r.get() for r in result_list ]
    pool.close()
    pool.join()
    print('Conditional Probability Matrix Finished in ' + str(np.round(time.time() - start_time,2)) + ' seconds.')
    return(result_list)

def get_cond_prob(sparse_peaks_mat, plot = True, num_slices = 16):
    '''
    Returns a 2d array of (d, dtheta)
    where the d values are incremented based on unique possible distances in the spatial slice
    and the dtheta is incremented by 1 degree.
    the values in the array correspond to Probability of DeltaTheta=dtheta given Distance=d
    '''
    # create list of slices from sparse peaks mat
    # for test methods
#     slices_list = [sparse_peaks_mat for i in range(10)]

    # the goal is two run the 8 cores twice, so divide the sparse matrix into 16 slices
    
    row_size = sparse_peaks_mat.shape[0] / num_slices
    col_size = sparse_peaks_mat.shape[1] / num_slices
    slices_list = []
    for i in range(num_slices):
        slices_list.append(
            sparse_peaks_mat[i*row_size:(i+1)*row_size, i*col_size:(i+1)*col_size, :])
    
    # get counts

    results = get_counts_multiprocessing(slices_list)
    joint_counts = sum(results)
    distance_counts = np.sum(joint_counts, axis = 1)
    cond_prob_mat = joint_counts / distance_counts[:,np.newaxis]
    
    if plot:
        plot_cond_prob(cond_prob_mat)
    return cond_prob_mat

def plot_cond_prob(cond_prob_mat):
    fig, ax = plt.subplots(figsize = (6,6))
    plt.imshow(cond_prob_mat.T, origin = 'lower')
    ax.set_ylabel('$\Delta \Theta$ (deg)')
    ax.set_xlabel('Distance (pxl)')
    plt.colorbar()
    plt.clim(0,np.percentile(cond_prob_mat,90))
    ax.set_title(r'P($\Delta\theta$ | d)')
    plt.show()         
    return

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
        dist, D, theta = stats._ksdist(cond_prob_mat[row,:], reference, ref_dist)
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

def to_discrete_pdf(cond_prob_mat):
    '''
    INPUT:
    cond_prob_mat = 2d numpy array with dimensions (d,dtheta) representing P(dtheta|d)
    OUTPUT:
    pdf_list = list of length d containing scipy.stats.rvdiscrete objects for P_d(dtheta)
    '''
    pdf_list = []
    logging.debug('running')
    dtheta = np.arange(cond_prob_mat.shape[1])
    for row in range(cond_prob_mat.shape[0]):
        try:
            pdf_list.append(rv_discrete(name=f'{row} pxls',values=(dtheta,cond_prob_mat[row,:])))
        except ValueError:
            # occurs when the probabilities don't add up to 1
            # This usuaully happens for the really large distances, so not that it matters much
            logging.debug(f'Row {row}')
            pdf_list.append(rv_discrete(name=f'{row} pxls blank'))
            
    return pdf_list

def expected_value(pdf_list):
    '''
    INPUT:
    pdf_list = list of length d containing scipy.stats.rvdiscrete objects for P_d(dtheta)
    --
    OUTPUT:
    ev = 1d array of length d with the expected values of each conditional probability distribution
    '''
    ev = np.zeros(len(test_pdf_list))
    for i,pdf in enumerate(test_pdf_list):
        if 'blank' in pdf.name:
            ev[i] = np.nan
        else:
            try:
                ev[i] = pdf.expect()
            except Exception as e:
                logging.debug(f'Row {i}')
                logging.error(traceback.format_exc())
                
    return ev