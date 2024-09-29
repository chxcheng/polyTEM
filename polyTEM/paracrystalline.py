"""
Built off of LatticeDisorder.py
"""
import numpy as np
import lmfit
import logging
import multiprocessing as mp
import time
import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rv_discrete


def _GaussianModel(q=0.235, qmin=0.20, qmax=0.32, sig=0.1, sigmin=0.01, sigmax=1):
    """
    Default Gaussian Model for peakfitting FFT Intensity vs q linecut
    """
    peak = lmfit.models.GaussianModel(prefix='peak_')
    pars = peak.make_params()
    pars['peak_center'].set(q, min=qmin, max=qmax)
    pars['peak_sigma'].set(sig, min=sigmin, max=sigmax)
    return peak, pars

def get_q_dist(q_datacube, peak_model, params, num_threads=8,  savefile=''):
    """
    From 4D datacube containing the Integrated Intensity vs q linecuts from FFT output,
    peak fit and extract q-value of peak center.
    Uses multiprocessing.
    ---
    INPUTS
    datacube = 4D np.ndarray with shape (x, y, q_index, 2), with the 4th dim being [qval, fft intensity]
    num_threads = integer for multiprocessing pool
    
    OUTPUTS
    q_list = flattened list with the fitted q_center at each (x,y) location
    """
    nrows, ncols, _, _ = q_datacube.shape
    logging.debug(f'datacube shape is {q_datacube.shape}')
    
    pool = mp.Pool(processes = num_threads)
    start_time = time.time()
    result_list = []
    for i in range(nrows):
        results = pool.apply_async(mp_peakfit_func,
                                   args = (q_datacube[i,:,:,:], peak_model, params))
        result_list.append(results)

    result_list = [r.get() for r in result_list]
    pool.close()
    pool.join()
    logging.info(f'Finished in {str(np.round(time.time() - start_time,2))}sec.')

    fit_results=np.array(result_list)
    logging.debug(f'mp function returns shape {fit_results.shape}')
    
    q_list = list(fit_results[:,:,0].flatten())
    if savefile:
        with open(savefile,'wb') as file:
            pickle.dump(q_list, file)
            
    return q_list
    
    
def mp_peakfit_func(row,model,pars):
    ncols = row.shape[0]
    result_row = np.zeros(shape=(ncols,2))

    for j in range(ncols):
        logging.debug(row.shape)
        q=row[j,:,0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") #Ignore RunTime Warning for Divide by nan or zero
            norm_counts = row[j,:,1] / np.nanmax(row[j,:,1])
        logging.debug(norm_counts.shape)
        logging.debug(q.shape)
        logging.debug(f'model is {model}')
        try:
            result = model.fit(norm_counts,pars,x=q, nan_policy='omit')
            result_row[j,0] = result.params['peak_center'].value
            result_row[j,1] = result.params['peak_fwhm'].value
        except TypeError as e: #input data less than number of fitting parameters
            logging.debug(f'Fitting at col = {j} failed due to too few input data (too many zeros or np.nan)')
            result_row[j,0] = np.nan
            result_row[j,1] = np.nan
    return result_row


def calc_g(qvals, label, N=0, plot=True, saveplotfile=''):
    """
    Bin/Histogram the distribution of qvals
    Calculate paracrystallinity % g where
    g = np.sqrt(E[d**2] / E[d] - 1)
    d == d-spacing of lattice plane
    E[x**n] == expected value, or nth moment of random variable x
    --
    INPUTS
    qvals = 1D np.array of q_centers to count
    label = plot title, can include spaces
    N = number of images this is binned across
    """
    hist = sns.histplot(qvals, stat='probability')
    plt.close()
    
    # get expected value
    d_spacings = [2*np.pi/patch.get_x() for patch in hist.patches]
    probs = [patch.get_height() for patch in hist.patches]
    RV = rv_discrete(name=label, values=(d_spacings,probs))

    # Expected values
    moment2 = RV.moment(2)
    mean = RV.moment(1)
    g = np.sqrt(moment2 / (mean**2) - 1)
    logging.debug(f'calculated g = {g}')

    # Plot
    if plot:
        plt.figure()
        plt.plot(d_spacings, RV.pmf(d_spacings))
        plt.xlabel('D spacing (A)')
        plt.ylabel('Probability Mass Function')
        plt.title(label)
        plt.text(s=f'Mean d-spacing={mean:.1f},\nParacrystalline g={g*100:.1f}%,\nCalculated across {N} Images',
                 x=mean+1, y=0.006)
        if saveplotfile:
            plt.savefig(saveplotfile, dpi=500, transparent=True, bbox_inches='tight')
    
    return g

def lattice_fluc(qvals, label):
    """
    Calculates the root-mean-square value of lattice parameter fluctuations
    e_rms = np.sqrt( E[e**2] )
    e = delta_d / d, where d=d-spacing
    E[x] is expected value of x
    """
    hist = sns.histplot(qvals, stat='probability')
    plt.close()
    
    # get expected value
    d_spacings = [2*np.pi/patch.get_x() for patch in hist.patches]
    d = np.mean(d_spacings)
    probs = [patch.get_height() for patch in hist.patches]
    RV = rv_discrete(name=label, values=(d_spacings,probs))
    
    e_func = lambda x: ((x-d)/d)**2
    e_rms = np.sqrt(RV.expect(e_func))
    return e_rms

def get_fwhm(x):
    """
    x is array of observed q-values
    assuming that distribution of x is Gaussian like
    the FWHM of the distribution of x is std*np.sqrt(8 * np.log(2))
    """
    fwhm = np.nanstd(x) * np.sqrt(8 * np.log(2))
    return fwhm

def get_sigma(fwhm):
    """
    assuming that the distribution is gaussian like
    then the standard deviation sigma = fwhm / (np.sqrt(8 * np.log(2)))
    """
    sigma = fwhm / (np.sqrt(8 * np.log(2)))
    return sigma
    
def get_fwhm_fromg(d, g, erms=0):
    """
    From Appendix B in Rivnay, et al. Warren-Averbach paper
    FWHM of a Voigt profile =
    (4*np.pi**2 * g**2 / d )(0.5346 + np.sqrt(0.2166 + (8*np.log(2) * erms**2)/g**2))
    """
    lorentz_component = 4 * (np.pi**2) * (g**2) / d
    gauss_component = 0.5346 + np.sqrt(0.2166 + (8*np.log(2) * erms**2)/(g**2))
    return lorentz_component * gauss_component


