import numpy as np
import mne

# handler to work with NaN values in an EEG recording
# got it from: http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y): 
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nan(input_signal):
    
    ne,ns  = input_signal.shape
    output_signal = np.zeros(input_signal.shape)
    for e in range(ne):

        nans, x = nan_helper(input_signal[e,:])        
        output_signal[e,:] = input_signal[e,:]
        output_signal[e,nans] = np.interp(x(nans), x(~nans), output_signal[e,~nans])        
    
    return output_signal

def create_mne_raw(signal,fs,ch_names,ch_types,description=None):
    
    ne,ns  = signal.shape    
    signal = interpolate_nan(signal)
    
    channel_names = ch_names
    channel_types = ch_types
    
    info = mne.create_info(channel_names, fs, channel_types)
    info['description'] = description
    
    raw = mne.io.RawArray(signal,info,verbose=False)    
    
    return raw    
    