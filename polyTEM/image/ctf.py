### ctf.py
"""
This module works with the output from CTFFIND4, which is used to identify the CTF
Module functions assist in ctf correction
"""
import pandas as pd
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from collections import defaultdict

class CTF:
    """
    Class storing the results of ctffind4 output
    
    Attributes:
        df (pd.DataFrame): Spatial frequency vs ctffit
        fit_summary (dict): defocus, astigmatism, phase shift, cross-correlation, and max fit
        fit_pars: ctffind4 input parameters: pixel size, acceleration voltage, spherical aberration,
                amplitude contrast, min. res., max. res., Box size, min. def., max. def.
        envelope_decay (float): beta parameter for envelope decay
    """
    
    # Attribute structures
    ctf_fit_df = pd.DataFrame(columns=['Micrograph Num','Spatial Frequency (1/A)', 'CTF fit','Signal'])
    summary_dict = {'defocus 1 (A)':0, 'defocus 2 (A)': 0, 'astigmatism':0,
                        'phase shift (rad)':0, 'cross-corr':0, 'max fit (A)':0}
    default_pars = {'Input file':'', 
                    'Pixel size':0, 'acceleration voltage':0, 'spherical aberration':0,
                    'amplitude contrast':0, 'min. res.':0, 'max. res.':0, 'Box size':0,
                    'min. def.':0, 'max. def.':0}

    def __init__(self, fit_df = ctf_fit_df, output = summary_dict, params = default_pars):
        self.df = fit_df 
        self.fit_summary = output
        self.fit_pars = params
        self.envelope_decay = None
                
        
    @classmethod
    def load(cls,filename,tilt_series=False):
        """
        Creates CTF instance from ctffind output file

        Args:
            filename (str): path to ctffind_output.txt

        Returns:
            ctf: CTF object

        """
        # Create CTF object
        ctf = CTF()
        info = 'summary'
        with open(filename) as file:
            logging.debug(f'ctf load: opened {filename}')
            for linenum,line in enumerate(file):
                if linenum == 0: # this is header line
                    logging.debug('skipping header line')
                    continue
                else:
                    ctf._process_fileline(line, info, linenum)
        
        avrot_fn = filename[:-4]+'_avrot.txt'
        info = 'avrot'
        with open(avrot_fn) as file:
            logging.debug(f'ctf load: opened {avrot_fn}')
            for linenum,line in enumerate(file):
                if linenum < 5: #identical to summary
                    continue
                else:
                    logging.debug(f'avrot file. actual line = {linenum}, linenum sent {linenum-4}')
                    ctf._process_fileline(line, info, linenum-4)
                    
        if tilt_series==True:
            # convert the dictionaries to dataframe
            ctf.fit_summary = pd.DataFrame.from_dict(ctf.fit_summary)
            
        return ctf
                    
                
                
    def _process_fileline(self, line, info='summary', linenum=None):
        """
        Processes ctffind4 output textfiles

        \b
        Expected info contanied in ctffind output (info=='summary'):
        Line 1: # CTF version, date and time
        Line 2: # Input file : <filename> ;
        Line 3: # <Parameter Name>: <Value> <Unit>;
        Line 4: # <Parameter Name>: <Value> <Unit>;
        Line 5: # Columns: #1 - micrograph number; #2 - defocus 1 [Angstroms]; #3 - defocus 2; #4 - azimuth of astigmatism; #5 - additional phase shift [radians]; #6 - cross correlation; #7 - spacing (in Angstroms);
        Line 6+: <Column Values separated by space>

        \b
        Expected info contained in ctiffind out _avrot:
        Line 1: # Output from CTFFIND version <version>, run on <date> <time>
        Line 2: # Input file: <filename>
        Line 3: # <Parameter Name>: <Value> <Unit>;
        Line 4: # <Parameter Name>: <Value> <Unit>;
        Line 5: # 6 lines per micrograph: #1 - spatial frequency (1/Angstroms); #2 - 1D rotational average of spectrum (assuming no astigmatism); #3 - 1D rotational average of spectrum; #4 - CTF fit; #5 - cross-correlation between spectrum and CTF fit; #6 - 2sigma of expected cross correlation of noise
        Line 6+: Values as described by Line 5.
        
        Assumes Python Version 3.7, where dictionaries are ordered!
        """
        if line[0] =='#':
            keys_list = re.findall(r' ([\w\. ]+):',line)
            values_list = [float(val_str) for val_str in re.findall(f'\d+\.?\d+',line)]
            if keys_list[0] == 'Columns':
                return
            else:
                self.fit_pars.update(dict(zip(keys_list,values_list)))
                return
        elif info == 'summary':
            values_list = re.split(' ',line)[1:]
            logging.debug(f'summary values are {values_list}.')
            if linenum == 5: 
                for key,val_str in zip(self.fit_summary.keys(),values_list):
                    self.fit_summary.update({key:float(val_str)})
            elif linenum >5: # then this is a tilt series
                if linenum == 6:
                    # only need to do this once to convert dict to defaultdict(list)
                    d = defaultdict(list)
                    for k,v in self.fit_summary.items():
                        d[k].append(v)
                    self.fit_summary = d
                    logging.info(f'converted to default dict!')
                    logging.debug(f' {self.fit_summary}')
                for key,val_str in zip(self.fit_summary.keys(),values_list):
                    self.fit_summary[key].append(val_str)
            return
        elif info == 'avrot':
            micrograph_num = linenum // 6
            linenum = linenum % 6
            if linenum == 1:
                values_list = [float(val_str) for val_str in re.split(' ',line)]
                self.df['Spatial Frequency (1/A)'] = values_list
                self.df['Micrograph Num'] = np.repeat(micrograph_num, len(values_list))
                return
            if linenum == 3:
                values_list = [float(val_str) for val_str in re.split(' ',line)]
                self.df['Signal'] = values_list
                return
            if linenum == 4:
                values_list = [float(val_str) for val_str in re.split(' ',line)]
                self.df['CTF fit'] = values_list
                return
            else:
                logging.debug(f'ctf.load does not process info/linenum={info}/{linenum}')
                return
        
    def _ctf_function(self,i:np.ndarray,j:np.ndarray,s,b,center_i,center_j):
        """
        Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4711343/
        
        To be used in np.fromfunction, in the form lambda i,j: _ctf_function(args)
        such that the lambda function is called with N parameters, where N is the rank of shape,
        and each parameter represents the coordinates of the array. 
        i.e. this function needs to work with i is a np.ndarray of row indices and j is a np.ndarray of col indices
        
        Args:
            i: row, pixel index 
            j: col, pixel index 
            s: size of array (assumes square)
            b: envelope decay parameter
        note that q=2pi/d (related to GIWAXS)
        where as u is 1/d
        
        Update 12.13.22 to allow for non-centered i and j.
        """
        w = self.fit_pars['amplitude contrast']
        wav = 1239.8/self.fit_pars['acceleration voltage'] # in picometers
        wav = wav/100 # from pm to A
        
        # what is the u value that corresponds to i and j
        du = 1/(s * self.fit_pars['Pixel size']) # units 1 / Angstrom
        x = j - center_j #this moves across columns j
        y = i - center_i #this moves vertically, across rows i
        logging.debug(f' x is {type(x)}: {x}')
        logging.debug(f' i is {type(x)}: {i}')
        u = np.sqrt(x**2 + y**2) * du # units 1/Angstrom
        theta = np.arctan2(y,x)
              
        Cs = self.fit_pars['spherical aberration']*1e7 # from mm to A
        Cs_abr = -0.5*np.pi*Cs * (wav**3) * (u**4)
        
        dz1 = self.fit_summary['defocus 1 (A)'] #dz1 is always greater than dz2
        dz2 = self.fit_summary['defocus 2 (A)']
        ast = np.deg2rad(self.fit_summary['astigmatism'])       
        dz = dz1*(np.cos(theta-ast))**2 + dz2*(np.sin(theta-ast))**2
        defocus_abr = np.pi * dz * wav * (u**2)
        
        gamma = Cs_abr + defocus_abr
        ctf = -np.sqrt(1-w**2)*np.sin(gamma) - w*np.cos(gamma)
        
        if b:
            envelope = np.exp(-b * u**2)
            ctf *= envelope
                       
        return ctf
    
    def build_array(self,s, plot=True):
        """
        uses the ctf fit output to build np.ndarray of input shape
        of the ctf function, to allow for correction.
        
        Args:
            s: size of square array
            
        Returns:
            shifted_arr: np.array with shape (s,s) containing CTF, with u=0 centered
        
        """
        b = self.envelope_decay
        center = int(s/2)
        array = np.fromfunction(lambda i,j: self._ctf_function(i,j,s,b,center,center),
                        shape = (s,s))
        
        shifted_arr = array
       
        if plot:
            plt.figure()
            du = 1/(s*self.fit_pars['Pixel size'])
            u_max = (s*du)/2
            plt.imshow(shifted_arr, cmap='RdBu',
                       vmin = -2, vmax = 2,
                      extent=(-u_max,u_max,-u_max,u_max))
            plt.colorbar()
            plt.xlabel('Frequency (1/A)')
            
            ub = 1/self.fit_summary['max fit (A)']+0.5
            plt.xlim([-ub,ub])
            plt.ylim([-ub,ub])
        
        return shifted_arr
      
    def wiener_filter(self,shape,snr):
        """
        build wiener filter for CTF, with size (s,s), 
        BUG: RIGHT NOW ASSUME SQUARE!
        and signal to noise ratio with the same size
        """
        ctf_array = self.build_array(s=shape[0],plot=False)
        logging.debug(f'Found nan values? {np.any(np.isnan(ctf_array))}')
        
        denom = np.square(ctf_array) + np.reciprocal(snr)
        logging.debug(f'Found nan values in denom?: {np.any(np.isnan(denom))}.')
        w = np.divide(ctf_array,denom)
        
        return w
    
    def _full_ctf_model(self, max_freq=None):
        """
        combines CTFFIND ctf fit result with envelope function.
        envelope fn = exp(-B * u**2)
        where u is the frequency vector and B is the decay rate
        Returns:
            model: lmfit Model instance to allow for envelope function fitting
        """
        freqs = self.df['Spatial Frequency (1/A)']
        ctf_vals = self.df['CTF fit']
        if max_freq:
            xmin=np.argmin(np.abs(freqs-0.08))
            xmax = np.argmin(np.abs(freqs-max_freq))
        ctf_fn = lambda x,B: np.multiply(ctf_vals[xmin:xmax],np.exp(-B * x**2))
        model = lmfit.Model(ctf_fn)
        return model
    
    def envelope_fit(self, min_freq=0.08):
        """
        Uses lmfit to fit the envelope parameter to the ctf
        Returns:
            result: lmfit ModelResult
            self.envelope_decay: Parameter B of envelope function, see _full_ctf_model
            
        """
        max_freq = 1/self.fit_summary['max fit (A)']
        if max_freq < min_freq:
            raise ValueError(f'CTFFIND output invalid, max frequency of ctffit is {max_freq}')
            
        xmax = np.argmin(np.abs(self.df['Spatial Frequency (1/A)'] - max_freq))
        xmin = np.argmin(np.abs(self.df['Spatial Frequency (1/A)'] - min_freq))
        model = self._full_ctf_model(max_freq = max_freq)
        pars = model.make_params(B=50)
        logging.debug(f'Envelope fitting y with shape {len(self.df["Signal"][xmin:xmax])} to x with shape {len(self.df["Spatial Frequency (1/A)"][xmin:xmax])}')
        result = model.fit(self.df['Signal'][xmin:xmax], params=pars,
                           x=self.df['Spatial Frequency (1/A)'][xmin:xmax],
                           nan_policy='omit')
        self.envelope_decay = result.params['B'].value
        return result
    
    def plot_lineout(self,q=False):
        freqs = self.df['Spatial Frequency (1/A)']
        ctf_sin = self.df['CTF fit']
        envelope = np.exp(-self.envelope_decay * freqs**2)
        ctf = envelope * ctf_sin
        fig = plt.figure()
        if q:
            plt.plot(2*np.pi*freqs,ctf)
            plt.xlabel('q (1/A)')
        else:
            plt.plot(freqs,ctf)
            plt.xlabel('Frequency (1/A)')
        plt.ylabel('CTF Intensity')
        return fig
                                

    
    
        
                         
        
            
                                      
            
            
                                 
        
                
            
            
            

    