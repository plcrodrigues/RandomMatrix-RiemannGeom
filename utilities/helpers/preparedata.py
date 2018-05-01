
import numpy as np
import mne

def _preprocess_data(raw, fparams):
    
    lfreq,hfreq = fparams
    
    # picking the channels of interest
    picks = mne.pick_types(raw.info, meg=False, eeg=True, 
                           eog=False, stim=False)
    
    # filtering the data
    raw.filter(lfreq, hfreq, method='iir', picks=picks, verbose=False)   
     
    return raw
    
def _slice_data(raw, tparams, events_interest):
    
    tmin,tmax   = tparams
    
    # get the events from stim channel
    events = mne.find_events(raw,verbose=False)

    # picking the channels of interest
    picks = mne.pick_types(raw.info, meg=False, eeg=True, 
                           eog=False, stim=False)

    # slice data into epochs    
    epochs = mne.Epochs(raw, events, events_interest, tmin, tmax,
                                 proj=True,
                                 picks=picks,
                                 add_eeg_ref=False,
                                 baseline=None,
                                 preload=True,
                                 verbose=False)    
    return epochs    
    