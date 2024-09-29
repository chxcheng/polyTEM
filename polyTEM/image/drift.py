### DRIFT CORRECTION
""" DRIFT CORRECTION CODE
Written by Camila Cendra, circa 2018

modified by Christina, 2023
"""

import numpy as np
import torch
import mrcfile
import matplotlib.pyplot as plt
import logging


        
def calculate(data_tensor):

    n_frames, x_size, y_size = data_tensor.shape
    max_size = max(x_size,y_size)
    pad = torch.nn.ConstantPad2d(padding=(0, max_size - y_size, 0, max_size - x_size), value=0)
    data_tensor = pad(data_tensor)


    x_center, y_center = max_size // 2, max_size // 2

    x_drift_list = [0]
    y_drift_list = [0]

    img_prev = data_tensor[0]

    for i in range(1, n_frames):
        img_curr = data_tensor[i]

        xc_results = calculate_xc(img_prev, img_curr, max_size)
        a, b = np.where(xc_results == np.amax(xc_results))
        logging.debug(f'Drift between frame #{i-1} and #{i}: {x_center - a}, {y_center -b}')
        x_drift_list.append(x_center - a[0])
        y_drift_list.append(y_center - b[0])

    return np.array(x_drift_list), np.array(y_drift_list)

def dose_study(data_tensor, verbose=True):
    n_frames, m, n = data_tensor.shape
    s = max(m, n)
    # Pad image. If m = n, no padding will be performed.
    pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
    data_tensor = pad(data_tensor)

    img_prev = data_tensor[0]
    xcc_list = []

    for i in range(1, n_frames):
        img_curr = data_tensor[i]

        xc_results = calculate_xc(img_prev.to('cuda'), img_curr.to('cuda'), s)
        xcc_list.append(np.sum(xc_results))
        print('xc intensity: ' + str(np.sum(xc_results)))

    return xcc_list


def calculate_xc(tensor1, tensor2, img_size, device='cuda'):
    fft_size = img_size
    fft_t1 = complex_fft(tensor1, fft_size, normalize=True)
    fft_t2 = complex_fft(tensor2, fft_size, normalize=True)

    prod = fft_t1 * np.conjugate(fft_t2)

    out = np.zeros((fft_size, fft_size, 2))
    out[:, :, 0] = np.real(prod)
    out[:, :, 1] = np.imag(prod)
    out_tensor = torch.from_numpy(out).to(device)

    xc = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(out_tensor), norm="forward"))[:,:,0]
    xc = tensor_shift_fft(xc).cpu().numpy()

    return xc


def correct_data(data, x_drift, y_drift, max_drift_allowed, save_array=''):
    """
    Corrects drift across frames in data
    
    Args:
        data: Image Dose-Fractionated Frames
        x_drift: List with drift in x-direction for each frame
        y_drift: List with drift in y-direction for each frame
        max_drift_allowed: int, threshold for drift tolerance
        save_array: filename to save final array
        
    Returns:
        data_corrected: data frames with drift corrected
        img_corrected: sum of data_corrected
    """
    logging.info('Correcting Data')
    padding = max_drift_allowed * 2
    n_frames, x_size, y_size = data.shape
    data_corrected = np.zeros((n_frames, x_size + 2 * padding, y_size + 2 * padding))

    # shift all pixels in each frame based on calculated drift value
    for i in range(n_frames):
        dx = x_drift[i]
        dy = y_drift[i]
        if np.abs(dx) <= max_drift_allowed and np.abs(dy) <= max_drift_allowed:
            data_corrected[i, (padding - dx):-(padding + dx), (padding - dy):-(padding + dy)] = \
                data[i, :x_size, :y_size]

    # trim edges of corrected frames to maintain resolution(dose), and flatten frames to single iamge
    trim = padding + max_drift_allowed
    data_corrected = data_corrected[:, trim:-trim, trim:-trim]
    img_corrected = np.sum(data_corrected, axis=0)
    logging.debug(f'Corrected Data has shape: {data_corrected.shape}')

    if save_array:
        np.save(save_array + '.npy', data_corrected)
        logging.debug('Corrected image array has been saved.')
        
    logging.info('Data Drift Corrected.')
    return data_corrected, img_corrected

        
def plot_2d_drift(x_drift, y_drift, dx=1.924, lines=False, savefile='', show_plot=True, file_type='.pdf'):
    '''
    by Camila
    '''
    n_frames = len(x_drift)

    fig = plt.figure(figsize=(5, 4))
    plt.scatter(x_drift * dx, y_drift * dx, s=20, c=np.arange(n_frames))
    range = [np.max(np.abs(x_drift)), np.max(np.abs(y_drift))]
    range_max = max(range) * dx * 1.3
    if lines:
        plt.plot(x_drift * dx, y_drift * dx, color='black', linewidth=0.25)
    plt.xlim([-range_max, range_max])
    plt.ylim([-range_max, range_max])
    plt.colorbar(label='frame #')
    plt.xlabel('Image drift in x̄ / Å', fontsize=14)
    plt.ylabel('Image drift in ȳ / Å', fontsize=14)
    if savefile:
        plt.savefig(savefile + file_type, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def complex_fft(tensor, s=5000, normalize=False):
    """Returns powder spectra of 2D tensor (image) using PyTorch implementation.
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
    tensor = tensor.double()
    m, n = tensor.shape
    pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)

    if normalize:
        tensor = tensor / torch.mean(tensor) - 1

    tensor = tensor * hanning(m)

    fft_tensor = torch.fft.fft2(pad(tensor).float(), norm="forward").cpu().numpy()
    return fft_tensor


def tensor_shift_fft(fft):
    """Shift zero frequency spatial frequency component to center of 2D image. For Pytorch implementation
    Args:
        fft: 2D FFT obtained using torch_fft function
    Returns:
        shifted FFT with DC frequency component in center of image.
    """
    m, n = fft.shape
    out = torch.cat((fft[-m // 2:], fft[:-m // 2]), dim=0)
    return torch.cat((out[:, -n // 2:], out[:, :-n // 2]), dim=1)


def hanning(N, device='cuda'):
    hanning_window = torch.from_numpy(np.outer(np.hanning(N), np.hanning(N))).to(device)

    return hanning_window


def read_mrc(fn):
    """Opens .mrc file containing single stack of images and returns numpy array.
    Args:
        fn: image filename
    Returns:
        np array of size (n, x, y)
        n is number of stacked images (generally 24)
        x, y is # pixels horizontally and vertically
    """
    mrc = mrcfile.open(fn, mode='r')
    img = np.flip(mrc.data, axis=1)
    mrc.close()

    return img.astype('float64')


def normalize_img(img_tensor):
    mean_tensor = torch.mean(img_tensor)

    return img_tensor / mean_tensor - 1

