import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import pandas as pd

import hdbscan
import sys
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
      
    Note:
        cluster_df will be deprecated in favor of DomainCollection
    
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
        print("Average cluster area (pxl) = " + str(np.mean(cluster_df['geometry'].area)))
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

 
def _domain_length(cluster, theta_type='lamellar'):
    '''
    INPUT:
    cluster: pandas Series, a row in cluster_df in which 
            ['theta'] is the orientation of crystalline stacking, w.r.t the horizontal
            ['geometry'] is a Shapely Polygon representing the domain geometry
    --
    OUTPUT:
    bb_Lc: length in direction of backbone chain, defined as longest line through polygon in that orientation
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

def domain_length_df(cluster_df, theta_type='lamellar', inplace = True):
    '''
    Get domain size statistics for each cluster in cluster dataframe.
    
    The lengths are returned for "backbone", "bbox_length", and "bbox_width".
    "backbone" length is the longest length through the polygon in the direction of backbone orientation, 
    we assuem that the backbone is perpendicular to the lamellar. "bbox_length" is the length of the bounding box
    in the backbone direction, and "bbox_width" is the length of the bounding box in the lamellar direction.
    
    '''
    Lc_df = cluster_df.apply(_domain_length,theta_type=theta_type, axis = 1, result_type = 'expand')
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


def get_domain_gradient(domain_array, center_index, center_orientation, plot=True):
    """
    Note: x,y follows i,j indexing such that x is iterating through the rows and y is iterating through the columns
    Thus, the parallel orientation is in the direction of y 
    and the perpendicular orientation is in the direction of x,
    and these follow the conventions of plt.imshow()
    BUT are inverse to the cartesian coordinates as usually plotted.
    """
    input_indices = np.argwhere(~np.isnan(domain_array))
    input_values = domain_array[~np.isnan(domain_array)].flatten()
    cmin = np.min(input_values)
    cmax = np.max(input_values)
    if plot:
        plt.figure()
        plt.imshow(domain_array, vmin=cmin, vmax=cmax)
        plt.title('Clustered Domain')
    
    ## INTERPOLATION
    interp = CloughTocher2DInterpolator(input_indices,input_values)
    X = np.arange(0,domain_array.shape[0])
    Y = np.arange(0,domain_array.shape[1])
    X, Y = np.meshgrid(X, Y, indexing='ij')  # 2D grid for interpolation
    full_theta = interp(X,Y)
    if plot:
        plt.figure()
        plt.imshow(full_theta,vmin=cmin,vmax=cmax)
        plt.colorbar()
        plt.title('Interpolated')
        
    ## ROTATE SUCH THAT NEMATIC DIRECTOR OF THE CENTROID IS PARALLEL TO ----> (YAXIS, USING IJ INDEX)
    rotation_angle = -center_orientation
    ## Naturally rotates about the center of the image 
    rotated_full_theta = ndimage.rotate(full_theta,rotation_angle,reshape=True, order=0,cval=np.nan) 
    rotated_full_theta += rotation_angle ## adjust values to correspond to the rotation
    if plot:
        plt.figure()
        plt.imshow(rotated_full_theta,
               vmin=cmin+rotation_angle,
               vmax=cmax+rotation_angle)
        plt.colorbar()
        plt.title('Rotated')
        
    ## TO EXTRAPOLATE THE GRADIENT, FILL IN THE NP.NAN WITH NEAREST NEIGHBORS 
    tree = KDTree(np.argwhere(~np.isnan(rotated_full_theta)))
    nan_indices = np.argwhere(np.isnan(rotated_full_theta))
    nn_dists, nn_indices = tree.query(nan_indices,k=1)
    for index,(i,j) in enumerate(nan_indices):
        nn_i,nn_j = tree.data[nn_indices[index]].astype(int)
        rotated_full_theta[i,j] = rotated_full_theta[nn_i,nn_j]
    if plot:
        plt.figure()
        plt.imshow(rotated_full_theta)
        plt.colorbar()
        plt.title('Extrapolated')
        
    ## CALCULATE GRADIENT
    # Change theta into cartesian coordinates,
    nx = np.sin(np.deg2rad(rotated_full_theta))
    ny = np.cos(np.deg2rad(rotated_full_theta))
    n = np.array([nx,ny])
    x = np.arange(0,rotated_full_theta.shape[0])
    y = np.arange(0,rotated_full_theta.shape[1])
    # Get gradient
    D = np.gradient(n,x,y, axis=(1,2))
    # Make it an array and flip first axes such that
    # the first axes indexes the components of the vector n,
    # the second axes indexes the cartesian direction of the gradient
    D = np.array(D).transpose((1,0,2,3)) 
    dnx_dy = D[0,1,:,:]
    dny_dx = D[1,0,:,:]
    if plot:
        plt.subplot(1,2,1)
        plt.imshow(dnx_dy)
        plt.colorbar()
        plt.title(r'$\nabla_yn_x$')
        plt.subplot(1,2,2)
        plt.imshow(dny_dx)
        plt.colorbar()
        plt.title(r'$\nabla_xn_y$')
    
    return D, rotated_full_theta