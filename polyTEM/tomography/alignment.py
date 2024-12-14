"""
Tomography Alignment using fiduciary markers

Uses the alignment procedure developed by Zhongqi Jing and Frederick Sachs, 1990
https://www-sciencedirect-com.stanford.idm.oclc.org/science/article/pii/0304399191900425?via%3Dihub
by solving linear equations
    
"""
import cv2
from skimage.filters import threshold_multiotsu
import numpy as np
import logging
from scipy.linalg import lu_factor,lu_solve
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
import dill 
import matplotlib.pyplot as plt
import jsonpickle.ext.numpy as jsonpickle_numpy
from .. import utilities


class AlignmentResult():
    """
    AlignmentResult object contains the parameters and results of aligning a stack of tilt images
    based on fiduciary marker locations
    
    Achieved by solving for X = R(PTU+D)
    where X is the initial image coordinates,
    R is the rotation matrix from alignment error
    D is the displacement matrix from alignment error
    P is the Projection, and T is the tilt matrix.
    
    Thus the final, aligned coordinates = R^-1 X - D
    which can be represented as M \cdot X.T, where M is the affine transformation matrix
    
    The algorithm also applies a manually inputed origin shift 
    since the Y-axis should coincide with the tilt axis. In future versions, 
    this origin shift should be saved with the class object.
    
    Attributes
    --
    beta_history: shape (num_iterations,num_images)
    dx_history: shape (num_iterations, num_images)
    dy_history: shape (num_iterations,num_images)
    marker_u: shape (num_markers)
    marker_v: shape (num_markers)
    marker_w: shape (num_markers)
    init_x: shape (num_images, num_markers)
    init_y: shape (num_images, num_markers)
    final_x: shape (num_images, num_markers)
    final_y: shape (num_images, num_markers)
    err: shape (num_iterations,)
    iterations: int
    affine_transform_mat: shape (num_images,2,3)
    num_images: number of images
    num_markers: number of markers
    tilt_angles: list of tilt angles in DEGREES, with matching order
    
    """
    
    def __init__(self,beta_history=None,dx_history=None,dy_history=None,
                 num_images=None,num_markers=None,
                 u=None,v=None,w=None,
                 init_x=None,init_y=None,
                 final_x=None,final_y=None,
                 err=None,iterations=None,
                 tilt_angles=None):
        self.beta_history=beta_history
        self.dx_history=dx_history
        self.dy_history=dy_history
        self.num_images=num_images
        self.num_markers=num_markers
        self.marker_u=u
        self.marker_v=v
        self.marker_w=w
        self.init_x=init_x
        self.init_y=init_y
        self.final_x=final_x
        self.final_y=final_y
        self.err=err
        self.iterations=iterations
        self.tilt_angles=tilt_angles
        return
    
    def save(self,filename, json=False):
        """
        """
        if json:
            jsonpickle_numpy.register_handlers()
            utilities.saveas_json(self,filename)
        else:
            with open(filename,'wb') as outp:
                dill.dump(self,outp)
        return

    
    @classmethod
    def load(cls,filename, json=False):
        """
        Load from pickle
        """
        if json:
            result = utilities.load_json(filename)
        else:
            with open(filename,'rb') as inp:
                result = dill.load(inp)
        return result
    
    def plot_history(self):
        plt.figure()
        plt.plot(np.abs(np.mean(self.dx_history,axis=1)),'b-',label=r'$\langle|\Delta x|\rangle$')
        plt.plot(np.abs(np.mean(self.dy_history,axis=1)),'m-',label=r'$\langle|\Delta y|\rangle$')
        plt.plot(np.mean(np.rad2deg(self.beta_history),axis=1),'g-',label=r'$\langle\beta (deg)\rangle$')
        plt.plot(self.err_history,'k--',label='Iteration Error')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Iteration Number')
        return
    
    
    def get_affine_transform_matrix(self,marker_indices=[0,1,2],origin_shift=(0,0)):
        """
        Returns the affine transformation matrix that accounts for the alignment error calculated in AlignmentResult,
        such that aligned image coordinates (x,y) = M * initial image coordinates
                
        Args:
            marker_indices: list of 3 indices to use to get affine transform, choose indices that form a large triangle, cannot calculate affine transform from a straight line.
            origin_shift: tuple (x,y) to shift the origin by. Usually necessary to keep image index and coordinates consistent
        Returns:
            M: matrix shape (num_images,2,3)
        """
        if len(marker_indices) != 3:
            raise ValueError(f'Affine Transformation requires 3 markers, but {len(marker_indices)} marker indices was given')

        src_x = self.init_x[:,marker_indices]
        src_y = self.init_y[:,marker_indices]
        dst_x = self.final_x[:,marker_indices] + origin_shift[0]
        dst_y = self.final_y[:,marker_indices] + origin_shift[1]
        src = np.float32(list(zip(src_x,src_y))) #shape (num_images, 2, 3)
        dst = np.float32(list(zip(dst_x,dst_y))) #shape (num_images, 2, 3)

        self.affine_transform_mat = np.ones(shape=(self.num_images,2,3))    
        for i in range(self.num_images):
            # get affine transformation
            self.affine_transform_mat[i,:,:] = cv2.getAffineTransform(src[i,:,:].T, dst[i,:,:].T)
        
        return self.affine_transform_mat
    
    @classmethod
    def align_markers(cls,x_coo,y_coo,tilt_angles,num_images,num_particles,method='explicit',
                  max_iter=50,tol=0.1):
        """
        Iterative Algorithm for fiduciary marker alignment

        Based on Jing and Sachs (1990)

        Args:
            x_coo: numpy array shape (num_images,num_particles)
            y_coo: numpy array shape (num_images,num_particles)
            tilt_angles: list of tilt_angles in radians
            num_images: int
            num_particles: int
            method: 'explicit' or 'lu' solve method

        Returns:
            AlignmentResult
        """
        iteration = 0
        beta,delta_x,delta_y,u,v,w = find_alignment_error(x_coo,y_coo,num_images,num_particles,tilt_angles,method)
        err = np.max([beta,delta_x,delta_y])

        result = cls(init_x=x_coo,init_y=y_coo,
                     num_images=num_images,
                     num_markers=num_particles,
                     tilt_angles=np.rad2deg(tilt_angles))
        beta_history =[beta]
        dx_history = [delta_x]
        dy_history = [delta_y]
        err_history = [err]

        while err > tol and iteration < max_iter:
            x_coo,y_coo = shift_markers(x_coo,y_coo,beta,delta_x,delta_y)
            beta,delta_x,delta_y,u,v,w = find_alignment_error(x_coo,y_coo,num_images,num_particles,tilt_angles,method)
            err = np.max([beta,delta_x,delta_y])
            logging.debug(f'Iteration number {iteration} has error {err} with alignment shift {beta},{delta_x},{delta_y}')

            err_history.append(err)
            beta_history.append(beta)
            dx_history.append(delta_x)
            dy_history.append(delta_y)
            iteration += 1
            if iteration >= max_iter:
                logging.warning(f'Algorithm reached max iteration limit of {max_iter} before convergence.')
        result.beta_history = np.array(beta_history)
        result.dx_history = np.array(dx_history)
        result.dy_history = np.array(dy_history)
        result.err_history = np.array(err_history)
        result.marker_u = u
        result.marker_v = v
        result.marker_w = w
        result.iterations=iteration
        result.final_x = x_coo
        result.final_y = y_coo
        return result
    
    def match_coo(self,coo,x_origin_shift=0,y_origin_shift=0):
        """
        Given coordinate from first image in stack, find the corresponding coordinates
        in all tilt images that should come from the same (u,v,w)
        """
        # apply origin_shift
        coo - [x_origin_shift,y_origin_shift]
        # find (u,v,w)
        feature_center = np.append(coo,1)
        aligned_center = np.dot(self.affine_transform_mat[0,:,:],feature_center)
        feature_uvw = np.append(aligned_center,0) #assume feature located at z=0
        feature_centers = []
        for image_ind in range(self.num_images):
            next_aligned_center = np.dot(proj_tilt_mat(np.deg2rad(self.tilt_angles[image_ind])),feature_uvw)
            inv_M = cv2.invertAffineTransform(self.affine_transform_mat[image_ind,:,:])
            feature_centers.append(np.dot(inv_M,np.append(next_aligned_center,1)) + [x_origin_shift,y_origin_shift])
        coords = [x.round().astype(int)for x in feature_centers]
        return coords


    
###################################################################################################
##
## ALIGNMENT MODULE HELPER FUNCTIONS
##
###################################################################################################

def find_markers(imgarray, num_markers:int, contrast:str='bright', 
                 kernel_size=25, kernel_step = 5,
                 max_iter=5, _recursion_count=0, debug=False):
    """
    Find gold fiduciary markers in image
    
    Relies on large contrast of gold to identify nanoparticles.
    Uses otsu's threshold, image opening, and kmeans clustering.
    
    Args:
        imgarray: numpy 2D array of image
        contrast: string signally whether the gold are the "bright" or "dark" spots in image
        num_markers: number of gold particles in image
        
    Returns:
        row,col locations of the markers
    """
    thresholds = threshold_multiotsu(imgarray)
    regions = np.digitize(imgarray, bins=thresholds)
    
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    opening = cv2.morphologyEx(regions.astype(np.uint8),
                              cv2.MORPH_OPEN,
                              kernel)
    if contrast == "bright":
        markers = (opening == len(thresholds)).nonzero()
    elif contrast == "dark":
        markers = (opening == 0).nonzero()
    else:
        logging.error(f'Invalid contrast, must be "bright" or "dark". Given {contast}')
        return
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
        compactness, labels, centers = cv2.kmeans(np.transpose(markers).astype(np.float32),
                                                  num_markers,
                                                  None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        row = centers[:,0]
        col = centers[:,1]
    except Exception as e:
        # perform recursion
        logging.warning(f'kmeans failed with input k {num_markers}, reducing kernel size')
        new_kernel_size = kernel_size - kernel_step
        if new_kernel_size <= 0 :
            logging.error(f'kmeans failed, kernel size has gone to 0')
            return None,None
        iter_num = _recursion_count + 1
        if iter_num > max_iter:
            logging.error(f'kmeans failed, hit max iteration of {max_iter}. Returning markers image')
            return None,None
        output = find_markers(imgarray, num_markers, contrast,kernel_size = new_kernel_size, 
                             _recursion_count = iter_num,debug=debug)  
        if debug:
            return output[0], output[1], output[2]
        else:
            return output[0], output[1]
    else:
        if debug:
            return row,col, markers
        else:
            return row,col

def sort_marker_labels(coo_list):
    """
    used to sort the outputs from find_markers so that the labels match for each marker
    
    find_markers uses kmeans, so the markers are returned in an inconsistent order between images.
    this function solves the assignment problem by using euclidean distance between the points as the cost matrix
    The first image in the input list is used as the reference (or the "worker") and 
    the subsequent images will be reordered (treated as the "jobs" in the assignment problem definition)
    
    Args:
        coo_list: list where each element of the list is a np.array of x and y coordinates. 
                The shape of each array is expected to be (num_markers,2)
    Returns:
        sorted_coo_arr: numpy array with shape (num_images,num_particles,2)
                    
    """
    ref_img = coo_list[0]
    sorted_coo_list = [ref_img]
    for img_num in range(1,len(coo_list)):
        img = coo_list[img_num]
        cost_matrix = euclidean_distances(ref_img,img)
        ref_ind, img_ind = linear_sum_assignment(cost_matrix)
        sorted_coo_list.append(img[img_ind])
    return np.array(sorted_coo_list)
                                   

                                
def find_alignment_error(x_coords,y_coords,num_images,num_particles,tilt_angles,method='explicit'):
    """
    Finds the rotation angle and image displacement by aligning markers
    
    Uses the alignment procedure developed by Zhongqi Jing and Frederick Sachs, 1990
    https://www-sciencedirect-com.stanford.idm.oclc.org/science/article/pii/0304399191900425?via%3Dihub
    by solving linear equations
    
    Args:
        x_coords: numpy array shape (num_images,num_particles), 
        y_coords: numpy array shape (num_images,num_particles,), 
        num_images: int
        num_particles: int,total number of gold particles
        tilt_angles: numpy array shape (num_images,) of tilt angles, in RADIANS
        method: 'explicit' or 'lu' solve
    """    
    ## let cartesian x refers to column position
    ## let cartesian y refers to row position
    ## Written for the structure  b = [-y(i0,js),x(i0,js),-y(i1,js),x(i1,js), ...]

    # build linear matrix array A
    partial_coeff_mat = np.zeros((num_images,2*num_particles,3*(num_images+num_particles)))
    partial_b = []
    matrixA = np.zeros((num_images,num_particles,3*(num_images+num_particles)))
    matrixB = np.zeros((num_images,num_particles,3*(num_images+num_particles)))
    for i in range(num_images):
        for j in range(num_particles):
            beta_index = i
            deltax_index = num_images+i
            deltay_index = (2*num_images)+i
            u_index = (3*num_images)+j
            v_index = (3*num_images)+num_particles+j
            w_index = (3*num_images)+(2*num_particles)+j
            
            # matrix A satisfies x*beta - delta_y - v = -y
            matrixA[i,j,beta_index] = x_coords[i,j]
            matrixA[i,j,deltay_index] = -1
            matrixA[i,j,v_index]=-1
            
            # matrix B satisfies y*beta + delta_x + cos(a)u + sin(a)w = x
            matrixB[i,j,beta_index] = y_coords[i,j]
            matrixB[i,j,deltax_index]=1
            matrixB[i,j,u_index]=np.cos(tilt_angles[i])
            matrixB[i,j,w_index]=np.sin(tilt_angles[i])
            
        # build coefficient matrix for linear equations
        partial_coeff_mat[i,:,:] = np.concatenate([matrixA[i,:,:],matrixB[i,:,:]])
        partial_b.append(np.concatenate([-y_coords[i,:],x_coords[i,:]],axis=0))
        
    # b = [-y(i0,js),x(i0,js),-y(i1,js),x(i1,js)]
    b = np.concatenate(partial_b,axis=0)
    coeff_mat = np.reshape(partial_coeff_mat,(2*num_images*num_particles,3*(num_images+num_particles)), order='C')
    logging.debug(f'coefficient matrix shape {coeff_mat.shape}')
    
    # solve linear equation
    if method == 'explicit':
        x,resid,rank,singular_val = np.linalg.lstsq(coeff_mat,b,rcond=None)
    elif method == 'lu':
        ATA = np.matmul(coeff_mat.transpose(),coeff_mat)
        ATB = np.matmul(coeff_mat.transpose(),b)
        lu, piv = lu_factor(ATA)
        x = lu_solve((lu, piv), ATB)
    else:
        logging.error(f'Invalid method {method}, must be "explicit"')
        return
    beta = x[0:num_images]
    delta_x = x[num_images:2*num_images]
    delta_y = x[2*num_images:3*num_images]
    u = x[3*num_images:(3*num_images)+num_particles]
    v = x[(3*num_images)+num_particles:(3*num_images)+(2*num_particles)]
    w = x[(3*num_images)+(2*num_particles):(3*num_images)+(3*num_particles)]
    return beta,delta_x,delta_y,u,v,w

def shift_markers(x_coords,y_coords,rot_angles,dx,dy):
    """
    Use approximate solution from find_alignment_errors to align markers
    
    Args:
        x_coords: numpy array shape (num_images,num_particles) 
        y_coords: numpy array shape (num_images,num_particles,)
        rot_angles: numpy array shape (num_images,) containing rotation alignment, in radians
        dx: numpy array shape (num_images,) containing shift in x
        dy: numpy array shape (num_images,) containing shift in y
    Returns:
        shifted_coords: numpy array shape (num_images,num_particles,2) containing 
    """
    num_images = x_coords.shape[0]
    num_particles = x_coords.shape[1]
    shifted_x_coords = np.zeros(x_coords.shape)
    shifted_y_coords = np.zeros(y_coords.shape)
    for i in range(num_images):
        beta = rot_angles[i]
        rot_mat = np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])
        rot_mat_inv = np.linalg.inv(rot_mat)
        displacement_mat = np.array([dx[i],dy[i]]).transpose()
        
        for j in range(num_particles):
            coords = np.array([x_coords[i,j],y_coords[i,j]]).transpose()
            new_coords = np.matmul(rot_mat_inv,coords) - displacement_mat
            shifted_x_coords[i,j] = new_coords[0]
            shifted_y_coords[i,j] = new_coords[1]

    return shifted_x_coords,shifted_y_coords


def chain_affine_transformation_mats(M0, M1):
    """ 
    Chaining affine transformations given by M0 and M1 matrices to use with OpenCV2 warpAffine.
    
    Written by Rotem on StackOverflow
    https://stackoverflow.com/questions/75388906/how-to-rotate-and-translate-an-image-with-opencv-without-losing-off-screen-data
    
    M0 - 2x3 matrix applying the first affine transformation (e.g rotation).
    M1 - 2x3 matrix applying the second affine transformation (e.g translation).
    The method returns M - 2x3 matrix that chains the two transformations M0 and M1 (e.g rotation then translation in a single matrix).
    """
    T0 = np.vstack((M0, np.array([0, 0, 1])))  # Add row [0, 0, 1] to the bottom of M0 ([0, 0, 1] applies last row of eye matrix), T0 is 3x3 matrix.
    T1 = np.vstack((M1, np.array([0, 0, 1])))  # Add row [0, 0, 1] to the bottom of M1.
    T = T1 @ T0  # Chain transformations T0 and T1 using matrix multiplication.
    M = T[0:2, :]  # Remove the last row from T (the last row of affine transformations is always [0, 0, 1] and OpenCV conversion is omitting the last row).
    return M

def tilt_mat(alpha,inverse=False):
    """
    Creates Tilt matrix for angle alpha, in radians
    """
    if not inverse:
        T = np.array([[np.cos(alpha),0,np.sin(alpha)],
                      [0,1,0],
                      [-np.sin(alpha),0,np.cos(alpha)]
                     ])
    elif inverse:
        T = np.array([[np.cos(alpha),0,-np.sin(alpha)],
                      [0,1,0],
                      [-np.sin(alpha),0,np.cos(alpha)]
                     ]) 
    return T

def proj_tilt_mat(alpha):
    """
    Matrix representing projection of tilted image 
    
    Args:
        alpha: angle in RADIANS
    """
    PT = np.array([[np.cos(alpha),0,np.sin(alpha)],
                   [0,1,0]
                  ])
    return PT