### Image object and functions
"""
Written by Christina Cheng, Adapted from Camila Cendra.
"""

import mrcfile
from PIL import Image as PImage
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from operator import itemgetter
import os
import copy
from scipy.ndimage import zoom

from . import drift, fft, ctf
from ..params import Parameters
from ..crystal_peaks import crystal_stack as peaks

class Image:

    """
    HRTEM Image
    
    Attributes:
        data (np.ndarray): image values
        dim (tuple): shape of data
        resolution (float): pixel resolution in Angstroms
        savedir (str): savefile location 
        _processor (str): 'cpu' or 'cuda:#'
        process_params (Parameter): dictionary
        pwd (np.ndarray): power spectrum of data
        ctf (CTF): Contrast Transfer Function information
    """

    def __init__(self,imarray,res,savedir='',params=Parameters(),
                processor='cuda:0', pwd=None, ctf=None):
        self.data = imarray
        self.dim = imarray.shape
        self.resolution = res
        self.savedir = savedir
        self.process_params = params
        self.processor(processor)
        self.pwd = pwd # Power spectrum of data
        self.ctf = ctf # Contains CTF Fit
    
    @classmethod
    def copy(cls, img, **kwargs):
        """
        creates a copy of Image instance, with optional kwargs
        to modify attributes. 
        
        Args:
            imarray: for data
            res: pixel resolution
            savedir: str
            pars: Parameters
            processor: str, 'cpu' or 'cuda:#'
            pwd: np.array power spectrum
            ctf: CTF object
        """
        im_array = kwargs.get('imarray',copy.copy(img.data))
        im_res = kwargs.get('res',copy.copy(img.resolution))
        im_savedir = kwargs.get('savedir',copy.copy(img.savedir))
        im_pars = kwargs.get('params',copy.deepcopy(img.process_params))
        im_processor = kwargs.get('processor',copy.copy(img._processor))
        im_pwd = kwargs.get('pwd',copy.copy(img.pwd))
        im_ctf = kwargs.get('ctf',copy.deepcopy(img.ctf))
        
        new_img = cls(imarray = im_array,
                      res = im_res,
                      savedir = im_savedir,
                      params = im_pars,
                      processor = im_processor,
                      pwd = im_pwd,
                      ctf = im_ctf)
        
        return new_img
        
    @classmethod
    def load(cls,filename,res,**kwargs):
        '''
        Creates Image object from image data filename.
        
        Args:
            filename: string containing image data
            res: pixel resolution of image data
            params: dictionary of parameters, optional key word argument
            savedir: string, save directory, optional keyword argument
            processor: string, optional keyword argument of either "cpu" or "cuda:#" 
            
        Returns:
            Image object
        '''
        _,ext = os.path.splitext(filename)
        if ext == '.mrc':
            mrc = mrcfile.open(filename, mode='r')
            imarray = mrc.data.astype('float64')
            mrc.close()
        elif ext == '.npy':
            imarray = np.load(filename, allow_pickle=True)
        else:
            img = PImage.open(filename)
            imarray = np.array(img).astype('float64')
            img.close()
            
        logging.debug(f'loaded imarray with shape {imarray.shape}')
        logging.debug(f'{type(imarray)}')
        pars = kwargs.get("params",Parameters())
        savedir = kwargs.get("savedir","")
        processor = kwargs.get("processor",'cuda:0')
        return Image(imarray,res, savedir=savedir, params=pars, processor=processor)
    
    @classmethod
    def _dummy_image(cls):
        """
        Creates empty Image instance
        """
        empty = Image(imarray=np.zeros(shape=(10,100,100)),
              res=1,
              savedir='',
              params=cls.default_params,
              processor='cuda:0')
        return empty
        
    def saveas_mrc(self):
        filename = os.path.join(self.savedir,f'{self.process_params["savefile_prefix"]}.mrc')
        with mrcfile.new(filename, overwrite=True)as mrc:
            mrc.set_data(self.data.astype('float32'))
        return   
            
            
    def update_params(self, params_dict):
        self.process_params.update(params_dict)
    
    def processor(self, punit = None):
        '''
        Change the processing unit for the Image object.
        
        Args: 
            punit: string, formatted as 'cpu' or 'cuda' (optional)
            
        Returns:
            prints the information of current proccessor after change. 
            
        '''
        if punit:
            if 'cuda' in punit:
                torch.cuda.set_device(punit)
                self._processor=f'{punit}'
                self.update_params({'processor':f'{punit}'})
            else:
                self._processor = 'cpu'
                self.update_params({'processor':'cpu'})
                
        logging.debug(f'Image Class set to allow {self._processor} use.')
        if 'cuda' in self._processor:
            logging.debug(f'CUDA running: {torch.cuda.is_available()}')
            logging.debug(f'Number of GPUs: {torch.cuda.device_count()}')
            logging.info(f'CUDA running on GPU #{torch.cuda.current_device()}, {torch.cuda.get_device_name()}')

    def view(self, figsize=None,save = False):
        '''
        Plots the image
        '''
        if len(self.dim) != 2:
            data = np.sum(self.data,axis=0)
            logging.debug(f'view: {self.data.shape}')
        else:
            data=self.data
        lb = np.percentile(data,10)
        ub = np.percentile(data,90)
        if figsize:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        plt.imshow(data, cmap='gray', vmin=lb, vmax=ub)
        plt.xlabel('Distance (pxls)')

        # Add scalebar
        hsize = 10 ** np.floor(np.log10(self.dim[1] * self.resolution * 0.2)) # round to nearest order of magnitude, in angstroms
        scalebar_size = round(hsize / self.resolution)
        scalebar = AnchoredSizeBar(plt.gca().transData, scalebar_size,
                                   f'{int(hsize/10)} nm', loc='lower right',
                                   frameon = False, size_vertical = 1, color='white', sep = 5)
        plt.gca().add_artist(scalebar)
        plt.show()

        if save:
            plt.savefig(f'{self.savedir}{self.process_params["savefile_prefix"]}_ImageView.png', dpi=500, transparent=True, bbox_inches='tight')
        return

    def flatten(self):
        """
        Flattens Image data into 2D
        """
        self.data = self.data.reshape(-1, self.data.shape[-1])
        self.dim = self.data.shape
        return
            
    def mask(self, mask, suffix='',inplace=False):
        """
        Apply mask to image
        
        Args:
            mask: np.array to apply
            suffix: str, to add to savefile prefix
            inplace: bool, if False, returns new Image instance
        Returns:
            None if inplace=True
            masked_img: Image object if inplace=False
        """
        # scale mask appropriately
        if mask.shape != self.dim:
            mask = zoom(mask,np.array([*self.dim]) / np.array([*mask.shape]))
                
        #apply mask
        if inplace:
            self.data *= mask
            if suffix:
                savefile_prefix = f'{self.process_params["savefile_prefix"]}_{suffix}'
                self.update_params({'savefile_prefix':savefile_prefix})
            return None
        else:
            masked_img = Image.copy(self,imarray=self.data * mask)
            if suffix:
                savefile_prefix = f'{self.process_params["savefile_prefix"]}_{suffix}'
                masked_img.update_params({'savefile_prefix':savefile_prefix})
            return masked_img

        
    def load_ctf(self,filename):
        """
        Get CTF object for this image based on CTFFIND output
        Runs the load method of the ctf model
        
        Args:
            filename: str, CTFFIND output .txt
        """
        self.ctf = ctf.CTF.load(filename)
        return
    
    def correct_ctf(self, snr_array):
        """
        Constructs a new Image() object with the same properties
        but contains ctf corrected data
        BUG: RIGHT NOW ASSUME SQUARE. only qorks with min!
        
        Arguments:
            snr_array: Signal-to-Noise Ratio array
        Returns:
            ctf_corrected_img: Image object
        """
        size = min(self.dim)
        # get image fft. Only interested in real portion
        real_fft, imag_fft = fft.fft(self.data, shape=(size,size), processor=self._processor, centered=True) 
        logging.debug(f'returned fft has shape {real_fft.shape} and is type {type(real_fft)}.')
        
        # get wiener filter
        # temporary assume that snr is array of 1
        wfilter = self.ctf.wiener_filter(shape=(size,size),snr=snr_array)
        if 'cuda' in self._processor:
            wfilter = torch.Tensor(wfilter).to(self._processor)
        logging.debug(f'Wiener filter is type {type(wfilter)}.')
        
        
        # correction is element wise multiplication
        logging.debug(f'Shape of real_fft{real_fft.shape}, shape of imag {imag_fft.shape}, wfilter {wfilter.shape}.')
        if 'cuda' in self._processor:
            fft_corrected = torch.multiply((real_fft+1j*imag_fft),wfilter)
        else:
            fft_corrected = np.multiply((real_fft+1j*imag_fft),wfilter)
        
        # return corrected image
        imdata_corrected = fft.ifft([fft_corrected],
                                   outshape=self.dim,
                                   processor = self._processor,
                                   centered_input=True,
                                   real_only=True)
        logging.debug(f'corrected img shape: {imdata_corrected.shape}.')
        if 'cuda' in self._processor:
            imdata_corrected=imdata_corrected.cpu().numpy()
        
        # create new Image with similar properties
        corrected_img = Image.copy(self, imarray=imdata_corrected)
        save_prefix = self.process_params['savefile_prefix'] + '_ctfcorrected'
        corrected_img.update_params({'savefile_prefix':save_prefix})
        return corrected_img
    
    def correct_drift(self, keep_frames=False):
        '''
        Corrects drift across image frames, GPU implementation ONLY.
        
        Args:
            keep_frames: boolean, determines whether to save drift-corrected data frames (True) versus only saving the summed data frames (False)
        Returns:
            corrected_img: Image instance containing drift-corrected data

        '''
        if self._processor == 'cpu':
            logging.error('Drift correction attempted on Image with processor set to  cpu')
            return
        
        if len(self.dim) != 3:
            logging.error(f'Drift Correction: Image object has incorrect dimensions. Expecting (#frames,x,y). Received: {self.dim}')
            return
               
        # calculate drift amount
        data_frames = torch.Tensor(self.data).to('cuda')
        self.x_drift, self.y_drift = drift.calculate(data_frames)
        data_frames = data_frames.to('cpu')
        torch.cuda.empty_cache()
        
        # plot drift values
        drift.plot_2d_drift(self.x_drift, self.y_drift, dx=self.resolution, lines=False, show_plot=True )

        # shifts frames by drift amount, and then sums all frames together to collapse into single image
        logging.info(f'Drift Correction: max_drift allowed {self.process_params["max_drift"]}.')
        data_frames_corr, img_corrected = drift.correct_data(self.data, self.x_drift, self.y_drift, self.process_params["max_drift"])
        
        corrected_img = Image.copy(self, imarray=img_corrected)
        save_prefix = 'driftcorrected'
        corrected_img.update_params({"savefile_prefix":save_prefix})
        
        if self.savedir:
            savefile = os.path.join(self.savedir,f'{self.process_params["savefile_prefix"]}_driftcorrected.npy')
            logging.info(f'Saving drift corrected numpy array to {savefile}')
            np.save(savefile, img_corrected)
            if keep_frames:
                savefile = os.path.join(self.savedir,
                                f'{self.process_params["savefile_prefix"]}_dataframes_driftcorrected.npy')
                logging.info(f'Saving drift corrected, dose-fractionated frames.')
                np.save(savefile, data_frames_corr)
                
        return corrected_img
    
    def power_spectrum(self, calculate=True, plot=False, **kwargs):
        """
        Get Power Spectral Density of image
        This current method requires gpu usage, and converts image data to torch.Tensor
        
        Args:
            calculate: boolean, if False then it just returns or plots the existing power spectral density.
            If calculate is False and there is no existing power spectral density, it will calculate anyways.
            plot: boolean.  If True, it will automatically attempt to save image
            q_contour_list: optional
            q_contour_color: optional
        Returns:
            self.pwd: np.ndarray containing power spectral density of Image
            qd
            counts
            plot: matplotlib figure if plot argument is True, otherwise returns False
        """
        if self._processor == 'cpu':
            logging.error('Power Spectrum requires gpu usage.')
            return
        if self.pwd is None or calculate is True:
            tensor = torch.Tensor(self.data).to(self._processor)
            self.pwd = (fft.tensor_psd(tensor,self.process_params['fft_size'])).cpu()  
        
        if plot:
            q_contour_list = kwargs.get('q_contour_list',[])
            q_contour_color = kwargs.get('q_contour_color','blue')

            if self.savedir:
                savefile=os.path.join(self.savedir,f'{self.process_params["savefile_prefix"]}_pwd.png')
            else:
                savefile = ''
                
            plot, (ax1, ax2) = plt.subplots(2,1) 
            ax_pwd = fft.plot(self.pwd, ax=ax1, s=self.process_params['fft_size'], 
                              dx=self.resolution,save_fig='',
                           q_contour_list=q_contour_list, q_contour_color=q_contour_color)
            q,counts,ax_ii = fft.q_lineout(self.pwd.to(self._processor), dx=self.resolution, plot=True, ax=ax2,
                                 savefile=savefile)
        else:
            q,counts,_ = fft.q_lineout(self.pwd.to(self._processor), dx=self.resolution, plot=False)
            
        return self.pwd, q,counts, plot
    
    def bandpass_filter(self,q_low=None,q_high=None,inplace=False):
        """
        Applies bandpass filter to image.
        
        Based on code originally written by Camila
        
        Args:
            q_low: lower bound for high pass
            q_high: higher bound for low pass
        Returns:
            corrected img: Image instance with bandpass filtered data
        """
        m, n = self.dim
        
        if q_low is None:
            q_low = self.process_params['q_low']
        if q_high is None:
            q_high = self.process_params['q_high']

        # Make raised cosine window to reduce boundary effects
        _, rc_window_m = _raised_cosine_window_np(m, beta=0.1)
        _, rc_window_n = _raised_cosine_window_np(n, beta=0.1)
        window = np.outer(rc_window_m, rc_window_n)   # window shape is (m, n)
        # Multiply real space image by window
        imarray = self.data * window
        # Pad image if shape[0] != shape[1]
        s = max(m, n)
        if m != n:
            # print('padding tensor to match row and col size ...')
            pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
            imarray = pad(imarray)
        
        corrected_imarray = fft.apply_fourier_filter(img=imarray,
                                                    dx = self.resolution,
                                                    fft_size = np.min(self.dim),
                                                    device = self._processor,
                                                    kind='bandpass',
                                                    q_low = q_low,
                                                    q_high = q_high)
        if 'cuda' in self._processor:
            corrected_imarray = corrected_imarray.cpu().numpy()
        #remove padding
        if inplace:
            self.data = corrected_imarray[:m,:n]
            return
        else:
            corrected_img = Image.copy(self, imarray = corrected_imarray[:m,:n])
            return corrected_img
    
    def fft_scan(self,pars=None):
        """
        Performs a sliding FFT scan across the image. Unlike featurize, it does not extract orientations
        and it does not apply a bandpass filter.  Equivalent to generating a virtual 4D-STEM.
        
        if pars is given, then pars will be used regardless of object instance attributes.
        
        Returns:
            fft_4d_datacube
        """
        # check if pars override
        if pars is None:
            pars = self.process_params
            
        fft_4d_datacube = fft.sliding_window_scan(self.data,
                                                 self.resolution,
                                                 pars['window_size'],
                                                 pars['fft_size'],
                                                 pars['step_size'],
                                                 pars['processor'],
                                                 angles=None,
                                                 orientation_filters=None,
                                                 q_vals = None,
                                                 plot_freq=0,
                                                 image_mask=img_mask)
        return fft_4d_datacube

    def featurize(self,kind='lamellar',pars=None, img_mask=None, plot_freq=1000):
        """
        extracts the orientations and q-centers of a specific periodic feature
        
        if pars is given, then pars will be reused, regardless of instance attributes.
        
        Returns:
            crystalStack
            q_datacube
        
        """
        # check if pars override
        if pars is None:
            pars = self.process_params

        # make fourier filters
        gaussian_filter = fft.gaussian_q_filter(pars['q'],
                                                pars['q_bandwidth'],
                                                pars['th_bandwidth'],
                                                pars['fft_size'],
                                                self.resolution)
        bandpass_filter = fft.bandpass_filter(pars['fft_size'],
                                              pars['q_low'],
                                              pars['q_high'],
                                              self.resolution)
        theta_filter = gaussian_filter * bandpass_filter
        qs = np.arange(pars['q_low'],pars['q_high'],pars['q_bandwidth'])
    
        # Set angles of interest
        if kind == 'lamellar' or kind == 'pi-pi':
            angles = np.arange(-90,90, step=1)
        elif kind == 'backbone':
            angles = np.arange(0,180,step=1)
        else:
            logging.warning(f'Input stacking type {kind} not recognized, defaulting to backbone')
            angles = np.arange(0,180,step=1)

        # perform sliding fft
        theta_datacube, q_datacube = fft.sliding_window_scan(self.data,
                                                             self.resolution,
                                                             pars['window_size'],
                                                             pars['fft_size'],
                                                             pars['step_size'],
                                                             pars['processor'],
                                                             angles,
                                                             theta_filter,
                                                             q_vals=qs,
                                                             image_mask=img_mask,
                                                             plot_freq=plot_freq)
        logging.debug(f'theta datacube of shape {theta_datacube.shape}')
        crystalStack = peaks.CrystalStack(img=None,datacube=theta_datacube)
        crystalStack.resolution = ((self.dim[0]*self.resolution)/theta_datacube.shape[0])/10 # nm
        crystalStack.q = pars['q']
        crystalStack.peaks_from_datacube()
        return crystalStack, q_datacube

    
########################################################################################
## HELPER FUNCTIONS
########################################################################################
def _raised_cosine_window_np(s, beta=0.2):
    """
    Generate 1D raised cosine window of length s and with roll-factor beta.
    :param s: pixel lenght of signal
    :param beta: roll-factor beta. Default=0.2 mainly zeroes down signal close to edges of image.
    :return:
         frequencies, window: frequency values and corresponding raised cosine signal
    """

    def rrc(f, b, T):
        """Finds raised cosine value for single frequency."""
        if f < (1.0 - b) / (2.0 * T):
            return 1.0
        elif f <= (1.0 + b) / (2.0 * T):
            return 0.5 * (1 + np.cos(np.pi * T / b * (f - (1 - b) / (2 * T))))
        else:
            return 0.

    T = 1 / s   # period
    frequencies = np.arange(-s / 2, s / 2, step=1)   # Distribute frequencies
    window = []
    for freq in frequencies:
        window.append(rrc(np.abs(freq), beta, T))   # Go over all frequencies

    window = window - np.amin(window)   # Set minimum to zero
    window = window / np.amax(window)   # Normalize so rc signal goes from 0 to 1

    return frequencies, window
            
            
    
                                      


        

        
        
