
"""
Adapted from Camila Cendra's nano package
"""

import numpy as np
import time
import torch
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2
from tqdm.autonotebook import tqdm
import logging


def fft(array, shape, processor, centered=True):
    """
    computes the fft based on the processor input.
    Wrapper around cpu_fft and tensor_fft
    
    Returns fft stored on the same processor as input
    """
    if 'cuda' in processor:
        real_fft, imag_fft = tensor_fft(torch.Tensor(array).to(processor), shape, centered)
    elif processor == 'cpu':
        real_fft, imag_fft = cpu_fft(array, shape, centered)
    else:
        raise ValueError(f'fft: input processor {processor} in valid. Only accepts "gpu" and "cpu"')        
        return
    return real_fft, imag_fft

def ifft(ffts:list,outshape,processor, centered_input=False, real_only=False):
    """
    Arguments:
        ffts: list object. if only one, then it should be treated as the total, complex fft.
                Otherwise, list should be [real, imag] components of fft
        processor: which processor to use
        centered_input: boolean which handles whether the fft is centered or not
        real_only: boolean which handles whether to output real or complex image.
                
    Returns:
        img: array-like. type depends on input processor
    
    """
    xsize,ysize=outshape
    # create complex fft, if given as separate real and imaginary
    if len(ffts)>1:
        fft = ffts[0] + (1j * ffts[1])
    else:
        fft = ffts[0]
    
    # calculate inverse fft, to produce complex image
    if 'cuda' in processor:
        if centered_input:
            fft = torch.fft.ifftshift(fft)
        img = torch.fft.ifft2(fft)
        if real_only:
            img = torch.view_as_real(img)[:,:,0]
    elif processor == 'cpu':
        if centered_input:
            fft = scipy.fft.ifftshift(fft)
        img = scipy.fft.ifft2(fft)
        if real_only:
            img = np.real(img)
    return img

def cpu_fft(img: np.ndarray, shape, centered=True):
    """
    Computes fft of 2D image stored on cpu using numpy
    Output fft is zero centered.
    
    Args:
        img: numpy.ndarray of image
        size: int shape of fft output size
        
    Returns:
        real: numpy array of real component of fft
        imag. numpy ndarray of imaginary component of fft
    """
    
    fft = scipy.fft.fft2(img, s=shape)
    if centered:
        fft = scipy.fft.fftshift(fft)
        
    real_fft = np.real(fft)
    imag_fft = np.imag(fft)
    return real_fft, imag_fft

def tensor_fft(tensor:torch.Tensor, shape, centered=False):
    """
    Computes fft of 2D tensor using pytorch GPU implementation
    TENSOR FFT IS NOT ZERO CENTERED BY DEFAULT!
    the comment out is legacy code from camila
    Args:
        tensor: 2D torch.tensor
        size: int shape of fft output size
        
    Returns:
        real: torch.Tensor of real component of fft
        imag: torch.Tensor of imaginary component of fft
    """
#     xsize,ysize = shape
#     # normalize tensor
#     tensor_norm = normalize_tensor(tensor)
#     # pad tensor with zeros function to get size desired
#     pad = torch.nn.ConstantPad2d(padding=(0, xsize - tensor.shape[1], 0, ysize-tensor.shape[0]),
#                                  value = 0)
#     padded_tensor = pad(tensor_norm)
    
#     # compute fft
#     fft = torch.view_as_real(torch.fft.fft2(padded_tensor))
    fft = torch.view_as_real(torch.fft.fft2(tensor.double(),s=shape))
    logging.debug(f'tensor_fft shape is {fft.shape}')
    real_fft = fft[:,:,0]
    imag_fft = fft[:,:,1]
    
    if centered:
        real_fft = torch.fft.fftshift(real_fft)
        imag_fft = torch.fft.fftshift(imag_fft)
    return real_fft, imag_fft

def tensor_ifft(fft):
    """
    inverse FFT using pytorch implentation for GPU use
    Arguments:
        fft: complex torch.Tensor
    
    Returns:
        
    """
    # torch is a general complex function, but we know our signal is real
    # legacy code assumes just working with real function instead of complex
    img = torch.view_as_real(torch.fft.ifft2(fft))
    return img

        
def tensor_psd(tensor, size=512):
    """
    Returns powder spectrum of 2D tensor (image) using PyTorch implementation.
    NOTE: location of operation (GPU or CPU) is determined by location of input tensor.
    Send tensor to GPU prior to using this function to perform operations in GPU (i.e. tensor.to(cuda))
    Args:
        tensor: 2D tensor (image)
        s: output size of FFT (s x s). tensor is padded with zeros prior to performing FFT operation
        to specified output size.
    Returns:
        fft: powder spectra (real^2 + complex^2) tensor of size (s x s) with Fourier Transform.
             DC frequency component is set in center of tensor.
    """
    start = time.time()
    
    # compute fft
    real_fft, imag_fft = tensor_fft(tensor, shape=(size,size), centered=False)
    
    # adding up real and imaginary components in FT
    psd = real_fft**2 + imag_fft**2
    
    # shift zero frequency to center of image
    shifted_psd = torch.fft.fftshift(psd)
    
    logging.debug(f'Tensor PSD Time:{np.round(time.time() - start,4)}')
    return shifted_psd

def normalize_tensor(tensor):
    """
    Tensor normalization operation. Tensor/mean - 1.
    """
    tensor = tensor / torch.mean(tensor) - 1
    return tensor

def plot(fft, s, dx, ax=None, q_contour_list=[], q_contour_color='blue', save_fig='', alpha=0.5):
    """Plots Fourier transform and optionally radial contours of q space.
    INPUTS:
    fft = np.ndarray
    s = size of fft
    dx = pixel resolution of base image (before fft)
    q_contour_list = list of q-values to be drawn as contours in figure
    q_contour_color = matplotlib color
    
    OUTPUT:
    fig = matplotlib figure 
    
   """
    if ax is None:
        fig, ax = plt.subplots()
    
    if len(q_contour_list) > 0:
        ax.imshow(fft, cmap='gray', vmax=np.percentile(fft, 99))
        for q in q_contour_list:
            f_pixels = get_q_pixels(q, s, dx)
            ax.add_patch(plt.Circle(((s-1) / 2, (s-1) / 2), f_pixels, facecolor='none',
                                    edgecolor=q_contour_color, alpha=alpha, linewidth=2, linestyle=':'))
            ax.annotate(str(np.round(q, 2)), xy=(s/2, s/2 + f_pixels), color=q_contour_color, alpha=alpha, fontsize=12)

        ax.plot()  # Causes an auto scale update.
    else:
        q_max = np.pi / dx
        ax.imshow(fft, cmap='gray', extent=[-q_max, q_max, -q_max, q_max], vmax=np.percentile(fft, 99))
        ax.set_xlabel('q / ${Å^{-1}}$')
        ax.set_ylabel('q / ${Å^{-1}}$')   
        if q_max < 1:  # Case of lower resolution than usual dx = 1.924
            ax.set_xlim([-q_max, q_max])
            ax.set_ylim([-q_max, q_max])
        else:
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])

    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight',transparent=True, dpi=500)
    
    return ax

def get_q_pixels(q, s, dx):
    """Conversion of reciprocal space component in angstroms to spatial frequency in number of pixels.
    Arguments:
        q: reciprocal space vector (2pi/d)
        s: size of Fourier transform
        dx: HRTEM image resolution in Angstroms/pixel
    Returns:
        f_pixels: q in pixel size
    """
    f_angstrom = q / (2 * np.pi) #this is equivalent to 1/d  
    freq_resolution = 1 / (s * dx)
    f_pixels = f_angstrom / freq_resolution

    return f_pixels

def q_lineout(fft_gpu, dx,q_bandwidth=0.002,q_increments=0.002,q0 = 0.1, qmax = 1.6, q_list=[], device='cuda',plot=False,
              ax=None, savefile=''):
    """
    Function performs azimuthal intensity integrations as function of q for selected q bandwidth.
    Args:
        fft_gpu: Pytorch tensor fourier transform
        dx: resolution in Angstrom / pixel
        q_bandwidth: halfwidth of bandpass filter
        q_increments: integration coarseness, ignored if q_list provided
        q0: q to start integration, ignored if q_list provided
        qmax: max q for q lineout, ignored if q_list provided
        q_list: list of values to calculate instead of construction based on q increments. q_list overrides other input
        
        device: cuda device
    
    Returns:
        x: np.ndarray of q values
        y: np.ndarray of counts, corresponding to q values
        plot: figure
    """
    x = []
    y = []
    M, M = fft_gpu.shape
    
    time_start =time.time()

    if not np.asarray(q_list).size:
        q_list = np.arange(q0,qmax,q_increments)
    for qi in q_list:
        bp_filter = torch.from_numpy(bandpass_filter(M, qi - q_bandwidth, qi + q_bandwidth, dx)).to(device)
        product = bp_filter.double() * fft_gpu.double() / torch.sum(bp_filter.double())
        x.append(qi)
        y.append(torch.sum(product).cpu().numpy())
    if plot:
        if ax is None:
            plot = plt.figure()
            ax=plt.gca()
        ax.plot(x,y)
        plt.xlabel('q (1/A)')
        plt.ylabel('Intensity')
        
        if savefile:
            plt.savefig(savefile,dpi=500,bbox_inches='tight',transparent=True)
            
    time_elapsed = time.time() - time_start
    logging.debug(f'q_lineout finished in {time_elapsed:.2f} seconds.')
    return np.array(x), np.array(y), plot

def bandpass_filter(s, q_low, q_high, dx):
    """
    Generates bandpass filter of size s x s for range of frequencies (q_low, q_high).
    
    Written by Camila
    
    Created bandpass filter is zero frequency centered
    Arguments:
        s: size in pixels of output bandpass filter(in fourier space)
        q_low: minimum q value in inverse Angstroms
        q_high: maximum q value in inverse Angstroms
        dx: HRTEM image resolution in Angstroms/pixel
    Returns:
        mask: numpy array size (M, M) with ones for pixels between selected bandwidth of q values and zeros elsewhere.
    """
    rows, cols = s, s
    crow, ccol = (s-1) / 2, (s-1) / 2

    f_low_pixels = get_q_pixels(q_low, s, dx)
    f_high_pixels = get_q_pixels(q_high, s, dx)

    mask = np.zeros((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= f_low_pixels ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= f_high_pixels ** 2))
    mask[mask_area] = 1

    return mask

def gaussian_q_filter(q, sigma_q, sigma_th, M, dx):
    """
    Generates gaussian filter in Fourier space for designated q center and bandwidth.
    
    Written by Camila
    
    Arguments:
        q: q vector in angstroms
        sigma_q: q bandwidth in angstroms
        sigma_th: theta bandwidth in degrees
        M: size of output fourier transform image
        dx: HRTEM image resolution in Angstroms/pixel
    Return:
        matrix: numpy array of size (NN, NN) with gaussian filter. Normalized by area (sum = 2).
    """
    f_pixels = get_q_pixels(q, M, dx)
    f_sigma_q_pixels = get_q_pixels(sigma_q, M, dx)
    f_sigma_th_pixels = get_q_pixels(sigma_th, M, dx)

    grid = np.linspace(-(M - 1) / 2, (M - 1) / 2, M)  # Array centering

    out = []  # temporary list to generate 2d output matrix

    for i in grid:
        a = 1 / (2 * np.pi * f_sigma_q_pixels * f_sigma_th_pixels)  # front in gaussian function
        # exponential term
        sub = ((grid - f_pixels) ** 2 / (2 * f_sigma_q_pixels ** 2) + (i) ** 2 / (2 * f_sigma_th_pixels ** 2))
        out.append(a * np.exp(-sub))

    matrix = np.array(out)
    matrix = matrix + ndimage.rotate(matrix, 180, reshape=False)
    matrix = matrix / np.sum(matrix)
    return matrix

def apply_fourier_filter(img,dx,fft_size,device='cpu',filter_arr=None,kind=None,**kwargs):
    """
    Perfomrs fourier filter
    
    Args:
        img: np.ndarray
        dx: Angstrom/pixel resolution of img
        fft_size: size of fft
        device: torch device to use
        filter_arr: filter array input, optional
        kind: type of filter to use if filter_arr is not given.
        kwargs: parameters to build filter of type _kind_
        
    
    """
    if filter_arr is None:
        if kind == 'bandpass':
            q_low = kwargs.get('q_low',None)
            q_high = kwargs.get('q_high',None)
            filter_arr = bandpass_filter(fft_size, q_low, q_high, dx)
            logging.debug('Applying bandpass filter')
        elif kind == 'gaussian':
            #gaussian filter
            q = kwargs.get('q',None)
            q_sigma = kwargs.get('q_sigma',None)
            th_sigma = kwargs.get('th_sigma',None)
            filter_arr = gaussian_q_filter(q, q_sigma, th_sigma, fft_size,dx)
    if 'cuda' in device:
        filter_arr = torch.Tensor(filter_arr).to(device)
    # apply filter to fft
    [real_fft,imag_fft] = fft(img,shape=(fft_size,fft_size),
                              processor=device,
                              centered=True)
    bp_fft = (real_fft + 1j*imag_fft) * filter_arr
    corrected_img = ifft([bp_fft],outshape=img.shape,
                         processor=device,
                         centered_input=True,
                         real_only=True)
    return corrected_img

        
def frc(fft1,fft2,dx,processor):
    """
    Fourier Ring Coefficient
    based on Baxter, et al.
    https://doi.org/10.1016/j.jsb.2009.02.012

    FRC(k) = \frac{\Sigma_k fft1(k) * conj(fft2(k)}
    {np.sqrt(\Sigma_k magnitude(fft2(k))**2) * np.sqrt(\Sigma_k magnitude(fft1(k))**2)}
    where k is the spectral frequency 
    k is equivalent to u in other modules, = 1/d
       
    BUG: ASSUMES SQUARE FFT of same shapes
    
    Arguments:
        fft1: [real, imag] components of fourier transform of image 1
        fft2: [real, imag] components of fourier transform of image 2

    Returns:
        FRC_u: np.ndarray of reciprocal space 1/d units
        FRC_counts: the FRC coefficient FRC(q)
    """
    if "cuda" in processor:
        # cross-power spectrum: fft1 * conjugate(fft2). Only take real portion
        real_comp = torch.mul(fft1[0],fft2[0]) #element wise multiply real components
        imag_comp = torch.mul(fft1[1], fft2[1]) # element wise multiply imag components, complex conjugate of fft2
        numerator = torch.add(real_comp,imag_comp) #element wise add


        # normalization by magnitude
        mag_fft1 = torch.add(torch.square(fft1[0]),torch.square(fft1[1]))   
        mag_fft2 = torch.add(torch.square(fft2[0]),torch.square(fft2[1]))
        denominator = torch.multiply(torch.sqrt(mag_fft1),torch.sqrt(mag_fft2))

        FRC = torch.divide(numerator,denominator)
    else:
        # cross-power spectrum: fft1 * conjugate(fft2). Only take real portion
        real_comp = np.multiply(fft1[0],fft2[0]) #element wise multiply real components
        imag_comp = np.multiply(fft1[1], fft2[1]) # element wise multiply imag components, complex conjugate of fft2
        numerator = np.add(real_comp,imag_comp) #element wise add

        # normalization by magnitude
        mag_fft1 = np.add(np.square(fft1[0]),np.square(fft1[1]))   
        mag_fft2 = np.add(np.square(fft2[0]),np.square(fft2[1]))
        denominator = np.multiply(np.sqrt(mag_fft1),np.sqrt(mag_fft2))

        FRC = np.divide(numerator,denominator)
        FRC = torch.from_numpy(FRC)
        
    # get q lineout
    # q lineout, input fft must be a tensor 
    #q = 2/pi, u. so if I want u in 
    s = fft1[0].shape[0]
    dq = (2*np.pi)/(fft1[0].shape[0]*dx)
    
    FRC_q,FRC_counts, _ = q_lineout(FRC, dx,q_bandwidth=dq, q_increments=dq, q0=0.05,qmax=2, device=processor)
    FRC_u = FRC_q/(2*np.pi)

    return FRC_u, FRC_counts

def ssnr(img1:np.ndarray, img2:np.ndarray, shape:tuple, dx:float, processor:str):
    """
    Spectral Signal to Noise Ratio
    based on Baxter, et al.
    https://doi.org/10.1016/j.jsb.2009.02.012
    
    SSNR(k) = \frac{FRC(k)}{1 - FRC(k)}
    where k is spectral frequency.
    
    This is used in creating the wiener filter for ctf correction
    So the output will be in u = 1/d spectral frequency and should have matching size as ctf
    
    BUG: These for loops are really slow lol but I can't bother thinking of making it faster right now
    
    Arguments:
        img1: np.ndarray
        img2: np.ndarray
        shape: tuple for output shape.
        dx: A/Pixel resolution of the input images
    Returns:
        ssnr_array: np array with input shape with signal to noise ratio 
    """
    fft1 = fft(img1, shape, processor, centered=True)
    fft2 = fft(img2, shape, processor, centered=True)
    logging.info('Calculating FRC')
    frc_u, frc_counts = frc(fft1,fft2,dx,processor) #returns as numpy regardless of processor

    ssnr = np.divide(frc_counts, (1-frc_counts))
        
    # Build SSNR array of shape (s,s)
    du = 1/(shape[0]*dx)
    u_bound = shape[0]*du/2
    x = np.linspace(-u_bound,u_bound,shape[0])
    u_x,u_y = np.meshgrid(x,x)
    u_z = np.sqrt(u_x**2 + u_y**2)

    logging.info('Making SSNR Array')
    ssnr_list=[]
    for target_u in tqdm(u_z.flatten()):
        ssnr_list.append(ssnr[np.argmin(np.abs(frc_u-target_u))])
    ssnr_array = np.array(ssnr_list).reshape(shape)
    return ssnr_array

def rotate_filter(single_filter, angles, device='cuda'):
    """
    Generates 3D tensor of size (M, M, number_angles) with stack of rotated Fourier filters to be used for
    datacube extraction.
    
    Args:
        single_filter: Initial filter as numpy array of size (M, M)
        angles: list or numpy array with rotation angles
        device: CUDA device send final output tensor
    Returns:
        tensor: 3D filters tensor with series of filters for azimuthal information extraction
    """
    size = single_filter.shape[0]
    filters_np = np.zeros((size, size, len(angles)))
    for i in range(len(angles)):
        filters_np[:, :, i] = ndimage.rotate(single_filter, angles[i], reshape=False)

    return torch.from_numpy(filters_np).to(device)

def sliding_window_scan(image,res:float, window_size:int, fft_size:int, step_size:int,device:str, angles:np.ndarray, orientation_filter:np.ndarray, q_vals:np.ndarray,image_mask,plot_freq:int):
    """
    Performs sliding psd across the input image. If orientation filter is given, then each local psd window ois converted to intensity vs orientation theta. Else, returns a 4d datacube of fft images.
    
    Built off of Camila's work. Requires use of GPU
    
    Args:
        image: np.ndarray
        res: image Angstrom/pixel resolution
        window_size: int, image window size
        fft_size: int, size of fft to calculate
        step_size: step size of sliding window scan
        device: 'cuda:#' for torch device
        angles: np.ndarray with orientation angles to probe
        orientation_filter: Gaussian filter that will be rotated to use for orientation analysis
        q_vals: np.ndarray of q-values to track intensity (i.e:np.arange(q0,qmax,dq))
        image_mask: mask to remove scans from final result if there are known defects in image (such as lacey carbon artifacts)
        plot_freq: Frequency of scan result to plot and show
        
        
    Returns:
        fft_4d_datacube: 4D np.ndarray containing the fft power spectrum for every (row,col)
        ii_datacube: 4D pyTorch tensor containing integrated intensity for every (row,col,theta)
        q_datacube: 4D np.ndarray containign the q_linecut counts for every (row,col,q)
        
    """
    # initialize datacubes
    num_rows = int((image.shape[0] - window_size) / step_size + 1)
    num_cols = int((image.shape[1] - window_size) / step_size + 1)
    

    orientation_filters_tensor = rotate_filter(orientation_filter, angles, device)
    ii_datacube = torch.from_numpy(np.zeros((num_rows,num_cols,len(angles)))).to(device)
    q_datacube = np.zeros((num_rows,num_cols,len(q_vals),2))

    
    # create hanning_window to reduce edge effects when performing fft
    hanning_window = np.outer(np.hanning(window_size), np.hanning(window_size))
    
    # sliding window scan
    # i iterates through rows of image
    # j iterates through cols of image
    # row, and col refer to the indexes of the output datacubes
    for row,i in enumerate(tqdm(range(0,image.shape[0]-window_size,step_size))):
        i_end = i+window_size
        for col,j in enumerate(range(0,image.shape[1]-window_size,step_size)):
            j_end = j+window_size
            img_slice = np.array(image[i:i_end,j:j_end])
            
            #handle mask, skip the window at boundaries of mask
            if image_mask is not None:
                mask_slice = image_mask[i:i_end,j:j_end]
                if np.any(mask_slice) and not np.all(mask_slice):
                    continue
            
            # perform fft 
            img_window = torch.from_numpy(img_slice * hanning_window).to(device).double()
            fft_power = tensor_psd(img_window,size=fft_size)
            
            # get theta orientations
            ii_theta = get_orientation_torch(fft_power,orientation_filters_tensor,device).cpu()
            ii_datacube[row,col,:] = ii_theta

            # get q intensities:
            q,counts,_ = q_lineout(fft_power, dx = res, q_bandwidth=0.005, q_list=q_vals,device=device)
            q_datacube[row,col,:,:] = np.array([q,counts]).T

            # plot for debug tracking
            window_num = row*num_rows + col
            if plot_freq != 0 and window_num%plot_freq==0:
                subplot_mini(img_window.cpu().numpy(), fft_power.cpu().numpy(), [angles, ii_theta], 
                             title=f'window={window_num}, row={row}, col={col}', dx=res)

                
            # end col code
        # end row code
    return ii_datacube,q_datacube

            
def get_orientation_torch(fft, filters, device):
    """ Gets Intensity tensor for different angles at any grid point (x,y). Uses broadcasting and torch
    operations to speed 2x process with respect to loop.
    
    Written by Camila
    
    :param fft: GPU torch tensor of fourier transform
    :param filters: GPU torch tensor of gaussian filters to be applied (different angles)
    :param device: CUDA device
    :return: intensity tensor at grid point x,y
    """
    m, n = fft.shape
    fft_broadcast = torch.empty(m, n, 1).to(device).double()
    fft_broadcast[:, :, 0] = fft

    intensity_theta = torch.sum(torch.mul(fft_broadcast, filters), [0, 1])

    return intensity_theta                
                                
def subplot_mini(image, fft_raw, I_vs_th, title='', dx=1.924, save_fig=''):
    """ Plots stack of figures to describe nano-FFT extraction process. From left to right,
    the following figures are plot: real space  nano-image, raw FFT, and filtered FFT.
    
    Written by Camila
    
    Args:
        image: real-space image
        fft_raw: calculated FFT
        fft_masked: FFT multiplied with bandpass filter
        title: optional title for first subplot
        dx: image resolution in Angstrom/pixel
        save_fig: filename in case user wants to save plot
    """
    q_max = np.pi / dx
    th, I = I_vs_th[0], I_vs_th[1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(2)
    fig.set_figwidth(10)
    # Plot nano image
    ax1.imshow(image, cmap='gray', extent=[0, image.shape[1]*dx/10, 0, image.shape[0]*dx/10],
              vmin=np.percentile(image,10),vmax=np.percentile(image,99))
    ax1.set_xlabel('nm')
    ax1.set_ylabel('nm')
    ax1.set_title(title, fontsize=10)
    # plot TEM nanoimage with window
    ax2.imshow(fft_raw, cmap='gray', extent=[-q_max, q_max, -q_max, q_max])
    ax2.set_title('FFT', fontsize=10)
    ax2.set_xlabel('${Å^{-1}}$')
    ax2.set_ylabel('${Å^{-1}}$')
    ax2.set_xlim([-1,1])
    ax2.set_ylim([-1,1])
    # plot I vs theta
    ax3.scatter(th, I, s=2, color='blue')
    ax3.set_xlabel('θ / degrees')
    ax3.yaxis.tick_right()
    ax3.set_ylabel('Counts / a.u. ')
    ax3.set_xticks(np.arange(th[0], th[-1]+2, step=45))
    ax3.yaxis.set_label_position('right')
    if save_fig:
        plt.savefig(save_fig, dpi=300)
    plt.show()
    
    
  
