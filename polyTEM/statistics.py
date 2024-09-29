## multiprocessing statistics
'''
These functions are used to help with bulk analysis of crystal stacks

Warning:
    This module will be deprecated in favor of the spatial module
'''
import numpy as np
import time
import sparse
import multiprocess as mp
import builtins
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pytem import spatial_analysis as spatial
from pytem.crystal_peaks import crystal_stack
from pytem.crystal_peaks import _conditional_probability as cp
from scipy.stats import uniform, linregress
from scipy.signal import correlate
from tqdm.auto import tqdm 
import logging
import torch



def get_stack_list(filenames_list, num_threads=8):
    '''
    Use multiprocessing to load stacks from filenames
    helps reduce time for large datasets
    '''
    start_time = time.time()
    result_list = []
    pool = mp.Pool(processes=num_threads)
    for filename in filenames_list:
        results = pool.apply_async(crystal_stack.CrystalStack.stack_from_file, args=[filename])
        result_list.append(results)
    
    result_list = [r.get() for r in result_list]
    pool.close()
    pool.join()
    
    end_time = np.round(time.time() - start_time,2)
    print(f'stack list created with {len(result_list)} stacks in {end_time} seconds')  
    return result_list

def get_spatial_df(stack_list):
    '''
    Uses multiprocessing to run function multiprocessing_df_func,
    which performs spatial analysis on each stack.
    All of the spatial analysis is them compiled into a dataframe
    '''
    start_time = time.time()
    result_list = []
    # check and reserve number of threads
    pool = mp.Pool(processes = 8)
    pbar = tqdm(total=len(stack_list)) #for progress bar tracking in jupyter
    for stack in stack_list:
        results = pool.apply_async(_mp_df_func, args=[stack], callback = lambda _: pbar.update(1))
        result_list.append(results)
        
    # retrieve results and close multiprocessing threads    
    result_list = [r.get() for r in result_list]
    pool.close()
    pool.join()
    pbar.close()

    
    # create summary dataframe
    df = pd.DataFrame(result_list, columns = ['location','q','moran','moran p','area','perimeter','backbone','bbox length', 'bbox width'])

    end_time = np.round(time.time() - start_time,2)
    print(f'Dataframe completed, {end_time} seconds.')
    return(df)
    
def _mp_df_func(stack):
    '''
    Given a TEM Image's CrystalStack Instance,
    Retrieve the average domain shape and Global Moran's I Statistics
    For use in Multiprocessing Function
    '''
    area = stack.outdir.split('/')[-3] 
    q = stack.outdir.split('/')[-2]
    
    cluster_df = spatial.create_polygons(stack.peaks_df, verbose=False)
    cluster_df = spatial.coherence_length_df(cluster_df, inplace=True)
    # I want total average cluster area
    cluster_size = np.mean(cluster_df['geometry'].area)
    cluster_length = np.mean(cluster_df['geometry'].length)
    bb_Lc = np.mean(cluster_df['backbone'])
    bbox_len = np.mean(cluster_df['bbox_length'])
    bbox_width = np.mean(cluster_df['bbox_width'])
    # moran value statistics
    try:
        moran = spatial.moran(cluster_df, var = 'complex_theta', verbose = False, plot=False)
        df_data = [area, q, moran.I, moran.p_sim, cluster_size, cluster_length, bb_Lc, bbox_len, bbox_width]
    except Exception as e:
        logging.warning(f'At Location {area}, Moran Failed with exception {e}')
        df_data = [area, q, np.nan, np.nan, cluster_size, cluster_length, bb_Lc, bbox_len, bbox_width]
        
    return df_data
    

def get_deltath_hist(filename, plot = True, fig = None, axs = None):
    '''
    Process and Plot the AngleOverlap histogram
    INPUTS
    filename = .csv output of the AngleOverlap.py script
    ---
    OUTPUTS
    delta_th_df = dataframe
    '''
    df = pd.read_csv(filename, index_col = 0)
    # correct the angle difference due to geometry
    # effectively always choosing the smallest angle difference between two directions
    # this is because orientations are bidirectional
    df['index_corr'] = [180 - np.abs(x) if np.abs(x)>90 else np.abs(x) for x in df['index']]
    # the correction method under counts delta_th = 0 and delta_th = 90
    # this because there are 4 ways to get a delta in [1,89] but two ways to get a delta = 0 or 90
    # (e.g.) delta = 1, -1, 179, and -179 will all become a delta = 1
    df.loc[df['index_corr'] == 0, 'delta_th'] *=2
    df.loc[df['index_corr'] == 90, 'delta_th'] *= 2
    if plot:
        if fig is None:
            print('Cannot plot without assigning figure')
            return
        if axs is None:
            print('Cannot plot without assigning axes')
            return      
        #plot corrected histogram
        axs.plot(range(0,91),df.groupby('index_corr').sum()['delta_th'],'-',alpha=0.5)
        
    return df

def z_test(x,y):
    """
    Standard Z-Statistic test
    Critical Z for p=0.01 is 2.33
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    z = (mu_x - mu_y) / np.sqrt(std_x**2 + std_y**2)
    return z

def fisher_z(rho):
    """
    Fisher Z Transform turns pearson correlation coefficient r
    into a normally distributed z value, such that
    z = arctanh(r)
    with mean 0.5*np.log((1+rho)/(1-rho))
    and stderr 1/np.sqrt(N-3)
    INPUTS:
    rho = np.ndarray of correlation coefficients
    """
    z = np.arctanh(rho)
    return z
    

def _ksdist(x, reference='uniform', ref_dist = None):
    """
    performs the kolmogorov-smirnov test against the uniform distribution
    returns the KS-distribution (comparison between the CDFs) and the D-value
    which is the max value in the KS-distribution. 
    --
    OUTPUT
    ksdist = np.array(shape=(x,1)) difference in CDF for each element in x
    D = float, the max difference in CDF among all values in x
    D_x = the element in x that has the max difference
    """
    # check inputs
    if reference == "uniform":
        ref_dist = np.linspace(uniform.cdf(0.01), uniform.cdf(0.99), len(x))
    elif ref_dist is not None:
        pass
    else:
        print(f'Error: {reference} reference not currently supported.')
        return
    
    # perform KS test
    ksdist = np.cumsum(x) - ref_dist
    D = max(ksdist)
    D_x = np.argmax(ksdist)
    return ksdist, D, D_x


def get_kstest(stack_list, reference="uniform",plot=True):
    """
    retrieves Kolmogorov-smirnov test D-value against all distances
    for each stack in stack_list
    --
    INPUTS
    stack_list = list of crystal stacks
    OUTPUTS
    kstest_list[sample][d_value_list] = list of all the ks_dvalues
    kstest_thetas[sample,distance] = theta with the largest CDF difference from uniform
    """
    kstest_list = []
    kstest_thetas = []
    fig = None
    if reference == "discrete uniform":
        rand_stack = crystal_stack.CrystalStack.random(n=stack_list[0].sparse_peaks_mat.nnz, dim=stack_list[0].sparse_peaks_mat.shape, img=None)
        rand_stack.conditional_prob(plot=False)
    else:
        rand_stack = None
    
    for stack in stack_list:
        stack.compare_to_random(reference, rand_stack, plot=False)
        kstest_list.append(stack.ks_dlist)
        kstest_thetas.append(stack.ks_thetas)
        
    if plot:
        fig = plt.figure()
        for dlist in kstest_list:
            plt.plot(dlist, alpha=0.5)
        plt.plot(np.nanmedian(kstest_list,axis=0), 'k-')
        plt.xlabel('Distance (pxl)')
        plt.ylabel('KS test D-value')
        
    return kstest_list, kstest_thetas, fig

def match_stack_lists(stack_list1, stack_list2=None):
    '''
    arbitrary CrystalStackLists may not having matching lengths and matching image orders.
    This function matches samples across the two lists.
    Matches returns the indices (i,j) for matched samples for stack list 1 and 2, respectively
    
    Assumes that stack_list1 is the shorter list
    '''
    areas1 = [crystalstack.outdir.split('/')[-3] for crystalstack in stack_list1]
    areas2 = [crystalstack.outdir.split('/')[-3] for crystalstack in stack_list2]
    matches = [(areas1.index(item),areas2.index(item)) for item in areas1 if item in areas2]
    
    return matches

def xcorr_slow(stack_list1, stack_list2=None,prematched=False):
    '''
    Across many samples, performs crosscorrelation of two 3-D datacubes that originate from the same sample.  
    First, checks that the datasets come match sample name
    Second, perform crosscorrelation (convolution of the datacubes) using scipy.signal.correlate
    This method is slow, tqdm will print progress
    --
    INPUT
    stack_list1: List of CrystalStack instances
    stack_list2: List of CrystalStack instances, if not inputted, then perform autocorrelation instead
    --
    OUTPUT
    xc_array: 4-D array of shape (len(matches), size[0]*2 - 1, size[1]*2 - 1, size[2]*2-1) 
                that represents (sample, lagx, lagy, lag_theta), centered around 0 lag
    ---
    KNOWN BUGS
    This uses A LOT of RAM space
    Takes about 5 seconds per pair of images.
    '''
    if stack_list2 == None:
        # If only one stack list is provided, perform autocorrelation instead.
        stack_list2 = stack_list1

    if len(stack_list1) < len(stack_list2):
        iter_list = stack_list1
        other_list = stack_list2
    else:
        iter_list = stack_list2
        other_list = stack_list1

    # match samples across the two lists
    if not prematched:
        matches = match_stack_lists(iter_list,other_list)
    else:
        matches = list(zip(np.arange(len(iter_list)),np.arange(len(iter_list))))

    # set up large cross correlation array
    size = iter_list[0].sparse_peaks_mat.todense().shape
    xc_array = np.zeros((size[0]*2 - 1, size[1]*2 - 1, size[2]*2-1))
    logging.debug(f'size = {size}, xc_array shape = {xc_array.shape}')

    for index,(i,j) in tqdm(enumerate(matches), total = len(matches)):
        # check that the filenames are the same
        x = iter_list[i]
        y = other_list[j]

        # logging.debug(f'loop #{index} for i={i},j={j}, x is on {x.outdir.split("/")[-3]} and y on {y.outdir.split("/")[-3]}.')
        # perform cross correlation
        in1 = x.sparse_peaks_mat.todense()/x.sparse_peaks_mat.todense().std()
        in2 = y.sparse_peaks_mat.todense()/y.sparse_peaks_mat.todense().std()
        xc = correlate(in1,in2)/in1.size
        logging.debug(f'xc shape = {xc.shape}')
        del in1,in2

        # save cross correlation to array
        xc_array += xc

    xc_array /= len(matches)
    return xc_array

def plot_xcorr(xc_array, dtheta = 0, resolution=3.17, vmin = 0, vmax = 0.015, norm='linear'):
    '''
    plotting function for cross correlation
    '''
    xc_slice = xc_array[:,:,dtheta + 179]
    
    limits = np.array([-(xc_slice.shape[0]-1)/2,(xc_slice.shape[0]-1)/2,
                       -(xc_slice.shape[1]-1)/2,(xc_slice.shape[1]-1)/2]) * resolution
    
    fig = plt.figure()
    if norm == 'linear':
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == 'log':
        if vmin == 0: vmin = 0.0001
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        logging.warning('plotting colors norm invalid, defaulting to linear')
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        
    plt.imshow(xc_slice, extent=limits, cmap='viridis',
               norm=norm)
    plt.xlabel('lag $\Delta x$ (nm)')
    plt.ylabel('lag $\Delta y$ (nm)')
    plt.xlim([-25,25])
    plt.ylim([-25,25])
    plt.title(f'lag $\Delta\Theta={dtheta}$')
    plt.colorbar()
    return fig
    
def remove_outliers(data, col, method='IQR'):
    '''
    removes outliers from pandas database using method input
    INPUT:
    data = pandas dataframe
    col = string of column name
    method = 'IQR', data more than 1.5*IQR outside the IQR are considered outliers.
             'mean', data more than 4 standard deviations from the mean are considered outliers.
    '''
    if method == 'IQR':
        Q1, Q3 = data[col].quantile([0.25,0.75])
        out_range = 1.5 * (Q3-Q1)
        mask = data[col].between(Q1-out_range, Q3 + out_range, inclusive='both')
        filtered_data = data.loc[mask]
    elif method == 'mean':
        mean = data[col].mean
        std = data[col].std
        mask = data[col].between(mean-4*std, mean+4*std, inclusive='both')
        filtered_data = data.loc[mask]
    else:
        logging.error(f'remove_outliers, method {method} not valid.')
        filtered_data = None
    return filtered_data
    
def xcorr_linecuts(arr, max_d):
    '''
    Outputs the radial lineout of average correlation.
    Borrows some helper functions from _conditional_probability (import as cp)
    INPUTS:
    arr = 2D correlation array, centered such that lag 0 is at center of array
    --
    OUTPUTS:
    linecut = 2D array with dimensions (d,2), average correlation as a function of radial distance d
    '''
    distance_mat = cp.make_distances_array(arr.shape[0], arr.shape[1])
    unique_distances = np.unique(distance_mat) # Given in Pixels
    # find_center
    if arr.shape[0] % 2 == 0 or arr.shape[1]%2==0: #array is even in shape?!
        logging.warning('Correlation Plot Dimensions are even? Double Check center!')
        
    x0 = np.floor(arr.shape[0]/2).astype(int)
    y0 = np.floor(arr.shape[1]/2).astype(int)
    
    dists = unique_distances[unique_distances < max_d]
    linecut = np.zeros(shape=(len(dists),2))
    linecut[:,0] = dists
    logging.debug(f'shape = {arr.shape}')
    for d_index, d in enumerate(dists):
        ii = 0
        logging.debug(f'd={d}')
        norm_counts = 0
        for shift_x, shift_y in np.argwhere(distance_mat == d):
            # add the intensity values for the radial distance d
            try:
                ii = ii + arr[x0+shift_x, y0+shift_y] + arr[x0-shift_x, y0-shift_y]
                norm_counts += 2
            except IndexError as e:
                logging.error(e)
                logging.error(f'array shape = {arr.shape}')
                logging.error(f'd={d}; x0={x0}; y0={y0}; shift_x={shift_x}; shift_y={shift_y}')
                return

        linecut[d_index,1] = ii / norm_counts
            
    return linecut

def overlap_density(stack_list1, stack_list2, reference='', xstal_area='all', savefile='',prematched=False):
    '''
    Compares overlap density of CrystalStacks peaks.
    Density defined by:
    (Area containing crystalline signal from list1 AND list2) / (Reference Area)
    
    INPUTS:
    stack_list1 = list of CrystalStack objects
    stack_list2 = list of CrystalStack objects
    reference   = determines reference area. options are None or 'sum'
        ''    = no reference area used imples that the reference area is the area of the full sized image
        'sum'   = uses the total crystalline area from list1 and list2 as the reference.
    xstal_area  = defines what type of crystalline area
        'all'   = anywhere (x,y) location that had fft peak, uses crystalStack.sparse_peaks_mat for coo
        'domains' = only (x,y) locations that belong to a cluster, uses crystalStack.peaks_df for coo
        'aligned' = only clusters that have aligned orientations between list 1 and list 2, modifies crystalStack.peaks_df for coo.
                    use with NO reference, result is then directly proportional to amount of interfacial area in the film
    --
    OUTPUTS:
    if overlap_density value is 0, then error occured. checking logging.ERROR
    '''
    # sort short and long lists in order to get all matches
    if len(stack_list1) < len(stack_list2):
        list_reversed = False # to keep track of reference order
        short_list = stack_list1
        long_list = stack_list2
    else:
        list_reversed = True
        short_list = stack_list2
        long_list = stack_list1        
    # match samples across the two lists
    if not prematched:
        matches = match_stack_lists(short_list,long_list)
    else:
        matches = list(zip(np.arange(len(short_list)),np.arange(len(short_list))))
        
    overlap_density = []

    for index,(i,j) in tqdm(enumerate(matches), total = len(matches)):
        # set so x pulls from the short list
        x = short_list[i]
        y = long_list[j]
        
        if xstal_area == 'all':
            try: 
                x_coo = x.sparse_peaks_mat.any(axis=2)
                y_coo = y.sparse_peaks_mat.any(axis=2)
                # Here, density is equivalent to normalized area (total image area = 1)
                both_density = sparse.elemwise(np.multiply,x_coo,y_coo).density
                if reference == 'sum':
                    ref_density = sparse.elemwise(np.add,x_coo,y_coo).density
                else:
                    ref_density = 1 # normalized total iamge area
            except Exception as e: #peaks_df == None of sparse_peaks_mat == None. Final overlap density output is 0
                logging.error(f'Overlap Density error for match index {index}, crystalStack not properly initialized.')
                logging.error(e)
                both_density = 0
                ref_density = 1
        elif xstal_area == 'domains':
            try:
                x_coo = np.array(list(x.peaks_df.coo.unique()))
                y_coo = np.array(list(y.peaks_df.coo.unique()))
                # overlap is number of duplicate coo points 
                # = (# of points in x) + (# of points in y) - (# of unique points in Union of XY)
                z = np.concatenate([x_coo,y_coo]) 
                xy_coo = np.unique(z,axis=0)
                both_density = z.shape[0] - xy_coo.shape[0] # number of duplicate (x,y) points
                if reference == 'sum':
                    ref_density = xy_coo.shape[0]
                else:
                    ref_density = x.sparse_peaks_mat.shape[0] * x.sparse_peaks_mat.shape[1]
            except Exception as e: #peaks_df == None of sparse_peaks_mat == None
                logging.error(f'Overlap Density error for match index {index}, crystalStack not properly initialized.')
                logging.error(e)
                both_density = 0
                ref_density = 1
        elif xstal_area == 'aligned': # this could be written recursively but *shrugs*
            try:
                aligned_overlap = sparse.elemwise(np.multiply,x.sparse_peaks_mat,y.sparse_peaks_mat) # from fft peaks, match for x,y, and theta
                aligned_coo = list(zip(aligned_overlap.any(axis=2).coords[0],aligned_overlap.any(axis=2).coords[1]))
                
                y_subset = y.peaks_df.loc[y.peaks_df['coo'].isin(aligned_coo)] # by taking subset of original peaks_df, constrained to points that are in clusters and match the coordinates               
                y_mod = y.peaks_df.loc[y.peaks_df['cluster_num'].isin(y_subset['cluster_num'])] # expand the subset to include all (x,y) points in the cluster, not just at the aligned point
                y_coo = np.array(list(y_mod.coo.unique())) # take just the coordinates
                x_subset = x.peaks_df.loc[x.peaks_df['coo'].isin(aligned_coo)]
                x_mod = x.peaks_df.loc[x.peaks_df['cluster_num'].isin(x_subset['cluster_num'])]
                x_coo = np.array(list(x_mod.coo.unique()))
                
                # effectively do what happens if xstal_area == 'domains':
                z = np.concatenate([x_coo,y_coo])
                xy_coo = np.unique(z,axis=0)
                both_density = z.shape[0] - xy_coo.shape[0]
                if reference == 'sum':
                    ref_density = xy_coo.shape[0]
                else:
                    ref_density = x.sparse_peaks_mat.shape[0] * x.sparse_peaks_mat.shape[1] #size of image
            except Exception as e:
                logging.error(f'Overlap Density error for match index {index}, crystalStack not properly initialized.')
                logging.error(e)
                both_density = 0
                ref_density = 1
        else:
            logging.error(f'overlap density: xstal_area {xstal_area} is not supported. Must be "all", "domains", or "aligned".') 
        ## end of if statements
        
        overlap_density.append(both_density / ref_density)
        
    if savefile:
        logging.debug(f'Saving file at {savefile}')
        np.savetxt(savefile, overlap_density)
        
    return overlap_density

def decay_length(expected_vals,distances,start_ind=0,end_ind=-1):
    y = np.log(np.abs(expected_vals[start_ind:end_ind]))
    x = distances[start_ind:end_ind]
    
    model = linregress(x,y)
    if np.isnan(model.slope) or np.isnan(model.intercept):
        logging.error("linear fit failed")
        return
    
    Ld = -1/model.slope
    
    return Ld, model.slope, model.intercept, model.rvalue

def decay_length_means(expected_val_list,distances,start_ind=0,end_ind=-1,plot=True):
    slope_list = []
    intercept_list=[]
    Ld_list = []
    
    if plot:
        fig=plt.figure()
        plt.xlabel('Distance (nm)')
        plt.title(r'log(<$2cos^2(\Delta\theta) -1$>)')
        
    for sample_num,ev in enumerate(expected_val_list):
        Ld, slope, intercept, r2 = correlation_length(ev,distances,start_ind,end_ind)
        slope_list.append(slope)
        Ld_list.append(Ld)
        intercept_list.append(intercept)
        
        if plot:
            plt.plot(distances[start_ind:end_ind],
                     np.log(np.abs(ev[start_ind:end_ind])),
                     'c-',alpha=0.2,linewidth=1)
            plt.plot(distances[start_ind:end_ind],
                     intercept + slope*distances[start_ind:end_ind],
                     'r--',alpha=0.2,linewidth=1)
        
    
    slope_mean = np.mean(slope_list)
    slope_std = np.std(slope_list)
    intercept_mean = np.mean(intercept_list)
    Ld_mean = np.mean(Ld_list)
    
    #remove outliers
    mask = np.abs(slope_list-slope) < 3*slope_std
    slope_list = list(compress(slope_list,mask))
    slope_mean = np.mean(slope_list)
    intercept_mean = np.mean(list(compress(intercept_list,mask)))
    stderr_of_mean = np.std(slope_list, ddof=1) / np.sqrt(len(slope_list))
    
    # Calcualte correlation decay length
    Ld = -1/slope_mean
    Ld_err = stderr_of_mean / slope_mean**2
    print(f'{L_d} +/- {Ld_err}')
    
    if plot:
        plt.plot(distances[start_ind:end_ind],
                 intercept_mean + slope_mean*distances[start_ind:end_ind],
                 'k-',alpha=1,linewidth=2)
        textstr = '\n'.join((
                f'slope = {slope_mean:.2f} +/- {stderr_of_mean:.2f}',
                f'$L_d$ = {Ld:.1f} +/- {Ld_err:.2f} (nm)'
            ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.25, textstr, transform=plt.gca().transAxes, fontsize=14,\
            verticalalignment='top', bbox=props)
        plt.xlabel('Distance (nm')
        plt.ylabel(r'$\ln\left(\langle 2 \cos^2(\Delta\Theta) - 1 \rangle\right)$')

    return
