import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import pandas as pd

import hdbscan
import sys
sys.path.append('/home/chxcheng/anaconda3/lib/python3.8/site-packages/alphashape-1.3.1-py3.8.egg')
import alphashape #using v 1.3
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiPoint # Using v1.8, do not switch to v2.0+, not compatible with alphashape 1.3
from shapely import affinity
import geopandas as gpd
import multiprocessing as mp

from libpysal import weights
import esda
import splot
from splot.esda import plot_moran, moran_scatterplot

import rasterio.features
import scipy.ndimage as ndimage
import torch
from pytem import cc_reduce_data as reduce
from tqdm.auto import tqdm
from scipy.signal import find_peaks

from pytem.crystal_peaks._process_datacube import _extend_signal, _remove_duplicates
from pytem.crystal_peaks import crystal_stack as peaks
from pytem import utilities

import warnings
from shapely.errors import ShapelyDeprecationWarning
import logging


def create_polygons(peaks_df, alpha = 1, verbose=True, num_threads=8, warning_msg="ignore"):
    '''
    create polygon for each cluster group using alpha shape set to concave hull
    alpha is used to determine the convexity and concavitiy of the final polygon
    first check if peaks_df["coo"] exists, otherwise create it
    
    BUG! 
    
    
    parallization using multiprocessing
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_msg, category=ShapelyDeprecationWarning)
        try:
            peaks_df["coo"]
        except KeyError:
            peaks_df["coo"] = list(zip(peaks_df.coo_x, peaks_df.coo_y))

        # create polygon for each cluster group using alpha shape set to concave hull
        cluster_df = pd.DataFrame(columns = ['cluster_num','theta','complex_theta','sin_orientation', 'geometry'])
        sub_df_list = []

        for index, cluster_num in enumerate(peaks_df["cluster_num"].unique()):
            sub_df_list.append(peaks_df[peaks_df["cluster_num"] == cluster_num])
        results = _make_polygons_multiprocessing(sub_df_list, peaks_df["cluster_num"].unique(), alpha, verbose, num_threads)

        cluster_df = gpd.GeoDataFrame(pd.concat(results), geometry = 'geometry')
        cluster_df['complex_theta'] = cluster_df['complex_theta'].astype(complex)

    if verbose:
        print("Average cluster area = " + str(np.mean(cluster_df['geometry'].area)))
    return cluster_df

def _make_polygons_multiprocessing(list_sub_df, cluster_num_list, alpha, verbose=False, num_threads=8):
    '''
    function used in create_polygons
    sets up the multiprocessing pool process to form alpha shapes
    collects multiprocessing results
    '''
    start_time = time.time()
    result_list = []
    pool = mp.Pool(processes = num_threads)
    for sub_df,cluster_num in zip(list_sub_df,cluster_num_list):
        logging.debug(f'mp function, make polygons, input arguments are #{cluster_num},a={alpha}')
        results = pool.apply_async(_multiprocessing_polygons_func,
            args = (cluster_num, sub_df, alpha))
        result_list.append(results)
    result_list = [r.get() for r in result_list]
    pool.close()
    pool.join()
    if verbose:
        print('Finished in ' + str(np.round(time.time() - start_time,2)) + ' seconds.')
    return(result_list)

def _multiprocessing_polygons_func(cluster_num, sub_df, alpha):
    '''
    creates an alphashape Polygon based on a given cluster, based on CrystalStack.peaks_df
    
    Args:
        cluster_num: integer, to organize clusters when multiprocessing Pool results are collected
        sub_df: CrystalStack.peaks_df containing points belonging to the same cluster
        alpha: alphashape parameter
    
    Returns:
        cluster_df: dataframe containing the cluster number, average orientation, and Polygon gemoetry
    '''
    cluster = list(sub_df["coo"])
    cluster_df = pd.DataFrame(columns = ['cluster_num','theta','complex_theta','sin_orientation','geometry'])
    try:
        polygon = alphashape.alphashape(cluster,alpha).buffer(0.2)
    except Exception as e:
        polygon = Polygon()
        logging.warning(f'Error creating polygon for cluster #:{cluster_num}')
        logging.warning(f'{e}')
        return cluster_df
    
    try:
        cluster_df.loc[cluster_num] = [cluster_num, np.mean(sub_df['theta']), np.mean(sub_df['complex_theta']), np.mean(sub_df['sin_orientation']), polygon]
    except ValueError:
        print('Error: MultiPolygon encountered for cluster #:' + str(cluster_num))
    return cluster_df

    
def moran(cluster_df, var = 'complex_theta', w_type = 'kernel', knn_size = 10, verbose = True, plot = True):
    '''
    Computes Moran's Global I Statistic
    cluster_df = GeoDataFrame()
    '''
    ## Set up weights between geometries based off of distance between domains
    if w_type == 'kernel':
        w = weights.Kernel.from_dataframe(cluster_df, geom_col = 'geometry')
    elif w_type == 'knn': 
        w = weights.KNN.from_dataframe(cluster_df, geom_col = 'geometry', k = knn_size)
    else:
        print('Error: Weight Type not recognized. Choose "kernel" or "knn"')
        return

    w.transform = 'R'
    moran = esda.Moran(cluster_df[var], w)
    if verbose:
        print('Morans I = %0.3f, with p-value %0.6f' % (moran.I.real, moran.p_sim.real))

    if plot:
        plt.figure()
        sns.kdeplot(moran.sim.real, color = 'gray',fill = True)
        plt.vlines(moran.EI_sim.real, ymin = 0, ymax = 1, colors = 'b')
        plt.vlines(moran.I.real, ymin = 0, ymax = 1, colors = 'r')
        plt.title('Reference Distribution')
        plt.xlabel('Moran I: ' + str(np.round(moran.I,3)))
        plt.show()

    return moran

def _moran_BV(cluster_df, x2_prefix, var = 'sin_orientation',w_type = 'kernel', knn_size = 10, verbose = True, plot = True):
    '''
    cluster_df = GeoDataFrame()
    '''
    if w_type == 'kernel':
        w = weights.Kernel.from_dataframe(cluster_df, geom_col = 'geometry', k = knn_size)
    elif w_type == 'knn': 
        w = weights.KNN.from_dataframe(cluster_df, geom_col = 'geometry', k = knn_size)
    else:
        print('Error: Weight Type not recognized. Choose "kernel" or "knn"')
        return

    w.transform = 'R'
    # temporarily since the x2 q_value can have multiple orientations, just pick 1.
    moran = esda.Moran_BV(cluster_df[var], cluster_df[x2_prefix+var], w)
    if verbose:
        print('Morans I = %0.3f, with p-value %0.6f' % (moran.I.real, moran.p_sim.real))

    if plot:
        plt.figure()
        sns.kdeplot(moran.sim.real, color = 'gray',fill = True)
        plt.vlines(moran.EI_sim.real, ymin = 0, ymax = 1, colors = 'b')
        plt.vlines(moran.I.real, ymin = 0, ymax = 1, colors = 'r')
        plt.title('Reference Distribution')
        plt.xlabel('Moran I: ' + str(np.round(moran.I,3)))
        plt.show()

    return moran

def moran_local(cluster_df, w_type = 'kernel', var = 'sin_orientation', knn_size = 10, plot = True):
    '''
    Computes Moran Local Statistic
    cluster_df = GeoDataFrame()
    '''
    if w_type == 'kernel':
        w = weights.Kernel.from_dataframe(cluster_df, geom_col = 'geometry')
    elif w_type == 'knn': 
        w = weights.KNN.from_dataframe(cluster_df, geom_col = 'geometry', k = knn_size)
    else:
        print('Error: Weight Type not recognized. Choose "kernel" or "knn"')
        return

    w.transform = 'R'

    if var == 'complex_theta':
        lmoran_real = esda.Moran_Local(np.real(cluster_df[var]), w)
        lmoran_imag = esda.Moran_Local(np.imag(cluster_df[var]), w)
        cluster_df['local I real'] = lmoran_real.Is
        cluster_df['local I imag'] = lmoran_imag.Is
        lmoran = (lmoran_real, lmoran_imag)
    else:
        lmoran = esda.Moran_Local(cluster_df[var], w)
        cluster_df['local I'] = lmoran.Is
                                  
    if plot:
        plot_lmoran(cluster_df, lmoran, var)

    return lmoran

def plot_lmoran(cluster_df, lmoran, var):
    # wrapper function
    if var == 'complex_theta':
        _plot_lmoran_complex_theta(cluster_df, lmoran)
    else:
        _plot_lmoran_sin_orientation(cluster_df, lmoran, var)
    return

def _plot_lmoran_complex_theta(cluster_df, lmoran):
    '''
    Adapted from https://geographicdata.science/book/notebooks/07_local_autocorrelation.html
    Plots local statistics, scatterplot quadrant, stastical significance, and cluster map
    '''
    #Set up figure and axes
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs = axs.flatten()

    # Plot domain orientations
    ax = axs[0]
    cluster_df.plot(column='theta', cmap='cet_CET_C3s', scheme='quantiles',
            k=18, edgecolor='white', linewidth=0.1, alpha=0.7, legend=True, ax=ax)
   
    spot_labels = [0,1,2]                            
     # Real, all horizontal #
    ax = axs[1]
    hor_sig = 1 * (lmoran[0].p_sim < 0.05)   
    # Imag
    vert_sig = 1 * (lmoran[1].p_sim < 0.05)
    spots = hor_sig + vert_sig
    labels = np.array([cluster_df['theta'][ind]*spot_labels[i] for ind,i in enumerate(spots)])
    labels[labels < 1] = np.nan
#     print(labels)
    cluster_df.assign(cl=labels).plot(column='cl', cmap='cet_CET_C3s', alpha=0.9,
                                      edgecolor='white', linewidth=0.1, ax=ax)

    for i, ax in enumerate(axs.flatten()):
        ax.set_axis_off()
        ax.set_title(['Theta Orientation', 
                      'Moran Cluster Map'][i], y=0)

    f.tight_layout()
    
    plt.show()
    
    
def _plot_lmoran_sin_orientation(cluster_df, lmoran, var = 'sin_orientation'):
    #Set up figure and axes
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    # Make the axes accessible with single indexing
    axs = axs.flatten()

                        # Subplot 1 #
    ax = axs[0]
    cluster_df.plot(column=var, cmap='Spectral', scheme='quantiles',
            k=5, edgecolor='white', linewidth=0.1, alpha=0.7, legend=True, ax=ax)

                        # Subplot 2 #
    ax = axs[1]
    q_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    labels = [q_labels[i-1] for i in lmoran.q]
    hmap = colors.ListedColormap([ 'red', 'lightblue', 'blue', 'pink'])
    cluster_df.assign(cl=labels).plot(column='cl',  \
            k=2, cmap=hmap, linewidth=0.1, ax=ax, \
            edgecolor='white', legend=True)

                        # Subplot 3 #

    ax = axs[2]
    sig = 1 * (lmoran.p_sim < 0.05)
    hmap = colors.ListedColormap(['grey','black'])
    labels = ['non-sig.', 'significant'] 
    labels = [labels[i] for i in sig]
    cluster_df.assign(cl=labels).plot(column='cl', categorical=True, \
            k=2, cmap=hmap, linewidth=0.1, ax=ax, \
            edgecolor='white', legend=True)


                        # Subplot 4 #
    ax = axs[3]
    hotspot = 1 * (sig * lmoran.q==1)
    coldspot = 3 * (sig * lmoran.q==3)
    doughnut = 2 * (sig * lmoran.q==2)
    diamond = 4 * (sig * lmoran.q==4)
    spots = hotspot + coldspot + doughnut + diamond
    spot_labels = [ '0 ns', '1 vertical hot spot', '2 isolated vertical', '3 horizontal hot spot', '4 isolated horizontal']
    labels = [spot_labels[i] for i in spots]
    hmap = colors.ListedColormap([ 'lightgrey', 'blue', 'lightblue', 'red', 'pink'])


    cluster_df.assign(cl=labels).plot(column='cl', categorical=True, \
            k=2, cmap=hmap, linewidth=0.1, ax=ax, \
            edgecolor='white', legend=True, alpha = 0.7)

    for i, ax in enumerate(axs.flatten()):
        ax.set_axis_off()
        ax.set_title(['Local Statistics', 
                      'Scatterplot Quadrant', 
                      'Statistical Significance', 
                      'Moran Cluster Map'][i], y=0)

    f.tight_layout()

    plt.show()
    

def _multiprocessing_domain_to_fft_func(cluster_geom, cluster_num, image, bp_filter, 
                                        img_dx, red_dimx = 231, red_dimy = 223, plot=False):
    '''
    For a given domain geometry in an image, get  the fft in for that domain only,
    and then filter to a specific q-range.
    --- INPUTS ---
    cluster_geom: Geometric Shapely Shape (Polygon)
    cluster_num: int, some identifying domain index
    red_dimx and red_dimy: size of the reduced image from the datacube process
        so that we can scale the cluster_geom to the size of the og image
    image: the original, drift corrected image
    bp_filter: bandpass filter to filter the resulting fft to a q_range of interest
    plot: Boolean
    '''
    # reduced data outshape is (231,223) original image shape is (3818,3690),
    xscale = image.shape[0]/223
    yscale = image.shape[1]/231
    mask=rasterio.features.rasterize([cluster_geom], out_shape=(223,231))
    mask=torch.from_numpy(ndimage.zoom(mask,(xscale,yscale), order = 0))

    # get FFT of the domain.
    rows, cols = np.where(mask != 0)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Covering areas without domains with noise of mean same intensity as image and 1std
    mean = np.nanmean(image[min_row:max_row, min_col:max_col])
    std = np.nanstd(image[min_row:max_row, min_col:max_col])

    product = image * mask
    domain = product[min_row:max_row, min_col:max_col]  
    noise = np.random.normal(mean, std, (domain.shape))

    # Add noise to regions where there is no domain (cover zeros with noise)
    domain += (domain == 0).double() * torch.from_numpy(noise)   

    # Get FFT
    fft = reduce.tensor_fft(domain, 512)
#   fft = bp_filter*fft

    # Plot examples
    if plot:
        # Plot domain section of image
        x_size = domain.shape[1] * img_dx/10
        y_size = domain.shape[0] * img_dx/10
        plt.imshow(domain, extent=[0, x_size, 0, y_size], cmap='gray', 
                   vmax=np.percentile(domain, 99))
        plt.clim([0,1000])
        plt.xlabel('distance / nm', fontsize=22)
        plt.ylabel('distance / nm', fontsize=22)
        plt.show()

        # plot fourier transform
        q_max = np.pi / img_dx
        plt.imshow(fft, cmap='gray', extent=[-q_max, q_max, -q_max, q_max], 
                   vmax=np.percentile(fft, 99))
        plt.xlabel('q / ${Å^{-1}}$')
        plt.ylabel('q / ${Å^{-1}}$')
        plt.show()
    if cluster_num%200 == 0:
        print('>', end = '')
    return (cluster_num, fft)
    
def _domains_to_fft(cluster_df, q_range, img_dx, driftCorrected_filename = 'image_drift_corrected.npy',
                  red_dimx = 231, red_dimy = 223):
    '''
    Uses multiprocessing to get the fft of all the domains in image listed by cluster_df
    returns a list of fft for each domain
    '''
    image = torch.from_numpy(np.load(driftCorrected_filename))
    image = torch.sum(image, dim=0).double()

    bp_filter = torch.from_numpy(
        reduce.bandpass_filter(M = 512, q_low=q_range[0],q_high=q_range[1], dx = img_dx))
    bp_filter = bp_filter.type(torch.FloatTensor)
    
    start_time = time.time()
    print('Getting FFT of individual domains for q_range=[%0.2f,%0.2f]' % (q_range[0], q_range[1]))
    result_list = []
    pool = mp.Pool(processes = 10)
    for cluster_num, cluster_shape in enumerate(list(cluster_df['geometry'])):
        plot = False
        results = pool.apply_async(_multiprocessing_domain_to_fft_func,
                                   args = (cluster_shape, cluster_num, image, bp_filter, img_dx, red_dimx, red_dimy,plot))
#         results = _multiprocessing_domain_to_fft_func(cluster_shape, cluster_num, image, bp_filter, img_dx, red_dimx, red_dimy,plot)
        result_list.append(results)    
    result_list = [r.get() for r in result_list]
    # sort list by cluster_num
#     result_list.sort(key = lambda tup:tup[0])
    pool.close()
    pool.join()
    print('Finished in ' + str(np.round(time.time() - start_time,2)) + ' seconds.')
    
    return result_list
    
def _peaks_from_domainfft(fft_list, q_val, img_dx, img_angles, q_sigma=0.01, fft_size=512,
                    threshold_func= lambda x: np.percentile(x,90)):
    print('Extracting Peaks from Domain FFTs')
    start_time = time.time()
    gaussian_filter = reduce.gaussian_q_filter(q_val, q_sigma, 0.03, fft_size, img_dx)
    bandpass_filter = reduce.bandpass_filter(fft_size, q_val - q_sigma, q_val + q_sigma, img_dx)
    selected_filter = gaussian_filter* bandpass_filter
    filters_list = reduce.make_gaussian_filter_tensor(selected_filter, img_angles, device = 'cuda')

    orientation_list = []
    for fft_tuple in fft_list:
        cluster_num = fft_tuple[0]
        fft_img = fft_tuple[1]

        m,n = fft_img.shape
        fft_broadcast = torch.empty(m,n,1).to('cuda').double()
        fft_broadcast[:,:,0] = fft_img

        # get angle dependence
        intensity_theta = torch.sum(torch.mul(fft_broadcast, filters_list), [0,1]).cpu()

        #find peaks
        I = _extend_signal(intensity_theta, int(1.5*180))
        peaks, _ = find_peaks(I, height = threshold_func(intensity_theta))
        peaks, _ = _remove_duplicates(peaks)
        peaks = peaks[np.argmax(I[peaks])]
        orientation_list.append(peaks)
        
        if cluster_num%200 == 0:
            print('>', end = '')

    print('Finished in ' + str(np.round(time.time() - start_time,2)) + ' seconds.')
    return orientation_list

def bivariate_moran(cluster_df, q_val, x2_prefix, img_dx, img_angles, q_sigma = 0.01, fft_size = 512,
                   var = 'orientation', w_type = 'kernel', plot = True):
    '''
    wrapper function
    '''
    # step 1: get ffts of the domains mapped to the original image
    fft_list = _domains_to_fft(cluster_df, q_range = [q_val - q_sigma,q_val + q_sigma], img_dx = img_dx)
    
    # step 2: extract peaks in these domain regions
    theta_list = _peaks_from_domainfft(fft_list, q_val = q_val, img_dx = img_dx, img_angles = img_angles,
                                              q_sigma=q_sigma, fft_size=fft_size)
    
    # step 3: add column to cluster_df
    cluster_df[x2_prefix + 'sin_orientation'] = [np.sin(np.radians(np.array(x))) for x in theta_list]
    cluster_df[x2_prefix + 'complex_theta'] = [np.exp(2j * np.deg2rad(np.array(x))) for x in theta_list]
    cluster_df[x2_prefix + 'theta'] = theta_list
    
    # step 4: calculate moran
    moranbv = _moran_BV(cluster_df, x2_prefix = x2_prefix, var = var, w_type = w_type, knn_size = 10, verbose = True, plot = plot)
    
    return moranbv


def _coherence_length(cluster, theta_type='lamellar'):
    '''
    INPUT:
    cluster: pandas Series, a row in cluster_df in which 
            ['theta'] is the orientation of crystalline stacking, w.r.t the horizontal
            ['geometry'] is a Shapely Polygon representing the domain geometry
    --
    OUTPUT:
    bb_Lc: coherenece length in direction of backbone chain, defined as longest line through polygon in that orientation
    ax_len: length of bounding box oriented in direction of backbone orientation (length == perpendicular direction to lamellar stacking)
    ax_width: width of bounding box oriented in direction of backbon orientation (width == direction of lamellar stacking)
    
    Note: The output is in the same units as the cluster geometry.  
    default unit is pixels, may need conversion!
    '''
    # Get oriented bounding box, oriented in the direction of crystallinity,
    # bounding box is rotated such that the backbone chain direction aligns with the x-axis
    if theta_type == 'lamellar':
        angle = cluster['theta'] - 90
    else:
        angle = cluster['theta']
    rotated_obj = affinity.rotate(cluster['geometry'],-angle,'centroid')
    rotated_box = rotated_obj.envelope
    
    # get length and width of the oriented bounding box
    try:
        xmin, ymin, xmax, ymax = rotated_box.bounds
        ax_len = xmax-xmin
        ax_width = ymax-ymin
        lengths=[]
        # to find length in direction of backbone orientation, we know the horizontal line must pass through a vertex
        for (vertex_x, vertex_y) in rotated_obj.exterior.coords:
            h_line = LineString([(xmin, vertex_y), (xmax, vertex_y)])

            # find the intersection of that horizontal line with the polygon, and get length
            lengths.append(rotated_obj.intersection(h_line).length)
        bb_Lc = max(lengths)
    except (ValueError,AttributeError):
        logging.warning(f'Bounding Box Failed for cluster {cluster["cluster_num"]}')
        ax_len = np.nan
        ax_width = np.nan
        bb_Lc = np.nan
    return ax_len, ax_width, bb_Lc

def coherence_length_df(cluster_df, theta_type='lamellar', inplace = True):
    '''
    Get the coherence length for each domain in the cluster dataframe
    Here, coherence length is defined as the length of the domain in the direction
    of crystallinity (as determined by theta)
    '''
    Lc_df = cluster_df.apply(_coherence_length,theta_type=theta_type, axis = 1, result_type = 'expand')
    Lc_df.columns = ['bbox_length','bbox_width','backbone']
    
    if inplace:
        cluster_df['backbone'] = Lc_df['backbone'].to_numpy()
        cluster_df['bbox_length'] = Lc_df['bbox_length'].to_numpy()
        cluster_df['bbox_width'] = Lc_df['bbox_width'].to_numpy()
        return cluster_df
    else:   
        return Lc_df
    
def plot_polygons_df(cluster_df, resolution = 3.13, fig=None, xlim=[], ylim=[]):
    color_palette = sns.color_palette('husl', 180, as_cmap=True)
    cpal = plt.cm.ScalarMappable(norm=colors.Normalize(0, 180), cmap=color_palette)

    if fig:
        plt.figure(fig)
    else:
        fig = plt.figure(figsize=(7,7))
        
    for index, row in cluster_df.iterrows():
        try:
            plt.fill(row['geometry'].exterior.xy[1],row['geometry'].exterior.xy[0],
                     color=cpal.to_rgba(row['theta']),alpha=0.3)
        except AttributeError: #"Multipolygon doesn't have exterior"
            # split the multipolygon
            for polygon in row['geometry'].geoms:
                plt.fill(polygon.exterior.xy[1], polygon.exterior.xy[0], 
                         color=cpal.to_rgba(row['theta']),alpha=0.3)
    plt.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 180), cmap=color_palette),label='Orientation (deg)', ax=plt.gca())
        
    ax = plt.gca()
    ax.set_xticks(ax.get_xticks(), labels=resolution * ax.get_xticks())
    ax.set_yticks(ax.get_yticks(), labels=resolution * ax.get_yticks())
    ax.set(xlabel='(nm)', ylabel='(nm)', xlim=xlim,ylim=ylim, aspect='equal')
    plt.gca().invert_yaxis()
   
    return fig


  
