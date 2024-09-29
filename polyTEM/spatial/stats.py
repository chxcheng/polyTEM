import numpy as np
from scipy.spatial import KDTree
from scipy.stats import linregress, rv_histogram
from collections import namedtuple
import logging

## Define Linear Model Result class that can be serialized
LinRegressModel = namedtuple("LinRegressModel", ['slope','intercept','rvalue'])

def extrapolate_NN(array):
    """
    Extrapolates the nematic orientations using Nearest Neighbors, to fill the bounding box
    """
    # build KDTree to rapidly look up nearest neighbors
    tree = KDTree(np.argwhere(~np.isnan(array)))
    nan_indices = np.argwhere(np.isnan(array))
    nn_dists, nn_indices = tree.query(nan_indices,k=1)
    extrapolated_data = np.copy(array)
    for index,(i,j) in enumerate(nan_indices):
        nn_i,nn_j = tree.data[nn_indices[index]].astype(int)
        extrapolated_data[i,j] = extrapolated_data[nn_i,nn_j]
    return extrapolated_data

def legendre_polynomial_2(dimension=2):
    """
    2nd-order Legendre Polynomial.
    dimension==2, 2D constrained is 2*(np.cos(np.deg2rad(x))**2) - 1
    dimension==3, 3D is 3/2*(np.cos(np.deg2rad(x))**2) - 1/2
    
    Returns:
        function
    """
    if dimension==2:
        return lambda x: 2*(np.cos(np.deg2rad(x))**2) - 1
    elif dimension==3:
        return (3/2)*(np.cos(np.deg2rad(x))**2) - (1/2)
    else:
        raise ValueError(f'Legendre polynomial dimension must be 2 or 3')
        

def exponential_decay_length(y,x,start_ind=0,end_ind=-1):
    """
    Fits an exponential to y by modeling log(y) = ax + b. Where the decay is given by -1/a
    
    Useful for modeling orientation correlation length where the input y is the
    expected value of the 2nd-order Legendre polynomial 
    
    Arguments:
        y,x: np.array
        start_ind, end_ind: int, indexes for fitting
        plot: boolean, if True, will plot over any existing figure! Does not create new figure.
    Returns:
        Ld: decay length
    """
    y = np.log(np.abs(y[start_ind:end_ind]))
    x = x[start_ind:end_ind]
    
    model = linregress(x,y)
    if np.isnan(model.slope) or np.isnan(model.intercept):
        logging.error("linear fit failed")
        return None, None
    
    Ld = -1/model.slope
    model = LinRegressModel(model.slope,model.intercept,model.rvalue) # convert class to namedtuple
    logging.debug(f'slope, intercept, and rvalue: {model.slope, model.intercept, model.rvalue}')
    
    return Ld, model

def make_histogram_distributions(array,bins):
    """
    Histogram values and return random variable distribution
    
    Arguments:
        array: 2D array with shape (n_samples,n_experiments). Histogram is performed by column.
    Returns:
        list of histogram distributions
    """
    hist_dist_list = []
    for dist in range(array.shape[1]):
        values = array[:,dist]
        hist = np.histogram(values[~np.isnan(values)], bins=bins)
        hist_dist_list.append(rv_histogram(hist))
    return hist_dist_list
    
    
def bending_probability(v1,v2,model="nematic", resolution=1, **kwargs):
    """
    Calculates the bending probability between aggregate segments v1 and v2
    Segments are quantified as v = (x,y,alpha,a,b,c)
    where (a,b,c) is a unit vector of the nematic director (segment tangent), located at (x,y)
    (x,y) are in units of pixels
    
    Models:
    1) Single-chain WLC in Free Solution
        p = exp(-1/2 * Lp * omega**2)
        Lp is the chain persistence length
        omega is the bending rate dtheta/dd (d is distance)
    2) WLC in thermotropic nematic field obtained from Frank-Oseen Theory
        p = exp(-bg)
        bg = Ks*(div(n))**2 + Kb*curl(div(n))**2
        bg = 0.5*Ks*(s1 + s2)**2 + 0.5*Kb*(b1**2 + b2**2)
        
    BUGS:
    RUNTIME WARNING for model='single-chain': invalid value encountered in arccos when calculating theta?
    
    Args:
        v1,v2: vector field of chain orientation segments
        model: "single-chain","Frank-Oseen" or "nematic"        
        resolution: A/pixel resolution, needed for model="single-chain"
        Lp: units A, Persistence Length of polymer, default 3nm
        Ks, Kb: units pixels
    """
    distance = np.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2)*resolution
    
    if distance == 0:
        # prevent nodes from being connected to itself. 
        p = 0
    elif model == "single-chain":
        Lp = kwargs.get('Lp',30) 
        theta = np.rad2deg(np.arccos(np.dot(v1[3:6],v2[3:6])))
        omega = theta/distance
        p = np.exp(-0.5*Lp*omega**2)
    elif model == "Frank-Oseen" or model == "nematic":
        Ks = kwargs.get('Ks',np.nan)
        Kb = kwargs.get('Kb',np.nan)
        s1,s2,b1,b2 = vector_curvature(v1[3:6],v2[3:6],v1[0:2],v2[0:2])
        g = 0.5*Ks*(s1 + s2)**2 + 0.5*Kb*(b1**2 + b2**2)
        p = np.exp(-g)
    return p

def vector_curvature(v1,v2,r1,r2):
    """ 
    Calculates the numerical components of curvature of v1(r1)
    using finite, single-sided (forward) difference between two unit vectors v1 and v2,
    whose origins are projected into the xy plane onto r1 and r2
    
    splay: s1 = dvx/dx, s2 = dvy/dy
    twist: t1 = -dvy/dx, t2 = dvx/dy
    bend: b1 = dvx/dz, b2 = dvy/dz
    
    These equations assume that the nematic vector is parallel to the z direction
    Assumed that r1!=r2
    
    Args:
        v1,v2: unit tangent vectors
        r1,r2: location of origin of v1 and v2
    Returns:
        s1,s2,b1,b2
    
    """
    ## First, rotate v1 to be parallel to z
    rot_mat = rotate_align_vectors(v1,[0,0,1])
    # rotate the frames and broadcast r into (x,y,z=0)
    rotated_v2 = np.matmul(rot_mat,v2)
    rotated_r1 = np.matmul(rot_mat,np.pad(r1,(0,1)))
    rotated_r2 = np.matmul(rot_mat,np.pad(r2,(0,1)))

    dvx = rotated_v2[0]
    dvy = rotated_v2[1]
    dx = rotated_r2[0] - rotated_r1[0]
    dy = rotated_r2[1] - rotated_r1[1]
    dz = rotated_r2[2] - rotated_r1[2]
    
    s1 = np.divide(dvx,dx,where=dx!=0)
    s2 = np.divide(dvy,dy,where=dy!=0)
    b1 = np.divide(dvx,dz,where=dz!=0)
    b2 = np.divide(dvy,dz,where=dz!=0)
    return s1,s2,b1,b2   

def rotate_align_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    
    Source: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def line_intersect(line1,line2):
    """find the intersecting points between two lines
    
    Args:
        line1 (array): array with shape (N,2) representing the X,Y values of the points of line 1
                        or array with shape (2,) representing the slope and intercept of line 1
        line2 (array): array with shape (N,2) representing the X,Y, values of the points of line 2
                        or array with shape (2,) representing the slope and intercept of line 2
    Returns:
        point (array): x,y
    """
    
    if len(line1.shape)>1:
        slope1 = (line1[1,1]-line1[0,1]) / (line1[1,0] - line1[0,0])
        intercept1 = line1[1,1] - slope1*line1[1,0]
    else:
        slope1 = line1[0]
        intercept1 = line1[1]
        
    if len(line2.shape) > 1:
        slope2 = (line2[1,1]-line2[0,1]) / (line2[1,0] - line2[0,0])
        intercept2 = line2[1,1] - slope2*line2[1,0]
    else:
        slope2 = line2[0]
        intercept2 = line2[1]
        
    x = (intercept2-intercept1)/(slope1-slope2)
    y = slope1*(x) + intercept1
    return np.array([x,y]) 

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