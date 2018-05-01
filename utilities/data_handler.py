import numpy as np
import pandas as pd
import mne

from helpers.getdata import _import_data
from helpers.getdata import get_bci_iv_2a
from helpers.preparedata import _preprocess_data, _slice_data

from numpy import array, empty_like, vstack, tile
from scipy.signal import filtfilt, butter
import pickle
import gzip

import collections

def import_data(path, subject=1, session=1, task=1):
    return _import_data(path, subject, session, task)            
    
def preprocess_data(raw, fparams):
    return _preprocess_data(raw, fparams)    
    
def slice_data(raw, tparams, events_interest):    
    return _slice_data(raw, tparams, events_interest)    

def get_bci_mi_dataset(subject, session=1):
    
    path = './datasets/bci-motor_imagery/'
    task = 1
    fparams = [8.0, 35.0] 
    tparams = [1.25, 3.75]
    
    raw, event_id = get_bci_iv_2a(path, subject, session, task)
    raw = preprocess_data(raw, fparams)    
    epochs = slice_data(raw, tparams, event_id)    
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    return X, y           

def get_alpha_waves_dataset(subject, session=1):
    
    def gen_windows(L, ns, step=1):
        return np.stack([w + np.arange(L) for w in range(0,ns-L+1, step)])

    filename = './datasets/alpha-waves/data.csv'
    data = pd.read_csv(filename, sep=';')
    fs = data['Sampling Rate'][0]    
    
    channels = collections.OrderedDict()
    channels['Channel 1'] = 'Fp1'
    channels['Channel 2'] = 'Fp2'
    channels['Channel 3'] = 'Fc5'
    channels['Channel 4'] = 'Afz'
    channels['Channel 5'] = 'Fc6'
    channels['Channel 6'] = 'T7'
    channels['Channel 7'] = 'Cz'
    channels['Channel 8'] = 'T8'
    channels['Channel 9'] = 'P7'
    channels['Channel 10'] = 'P3'
    channels['Channel 11'] = 'Pz'
    channels['Channel 12'] = 'P4'
    channels['Channel 13'] = 'P8'
    channels['Channel 14'] = 'O1'
    channels['Channel 15'] = 'Oz'
    channels['Channel 16'] = 'O2'
    
    nc = len(channels)
    ns = len(data)         
    X = np.zeros((nc,ns)) 
    for i,chname in enumerate(channels.keys()):
        X[i,:] = data[chname]
    
    chg = np.diff(data['CH_Event'])
    chg = np.where(chg > 0)[0]
    y = np.zeros((ns))
    
    aux = 1
    for i in range(len(chg)-1):
        interval = chg[i] + np.arange(chg[i+1]-chg[i])
        y[interval] = aux
        aux = (aux+1)%2
    
    signal = np.concatenate((X,y[None,:]))    
    channel_names = channels.values() + ['STIM']
    channel_types = ['eeg' for _ in range(nc)] + ['stim']
    info = mne.create_info(channel_names, fs, channel_types)
    raw = mne.io.RawArray(signal,info,verbose=False) 
    raw.notch_filter(freqs=50)
    raw.filter(l_freq=8, h_freq=35)
    raw.resample(fs/4)
    fs = raw.info['sfreq']
    
    picks_eeg = mne.pick_types(raw.info, eeg=True)
    picks_stm = mne.pick_types(raw.info, stim=True)
    start,stop = (raw.time_as_index([30])[0], raw.time_as_index([120])[0])
    X,t = raw.get_data(picks=picks_eeg, start=start, stop=stop, return_times=True)
    t = t - 30
    y = raw.get_data(picks=picks_stm, start=start, stop=stop).T    
    
    L = 128  
    st = L/4
    Xt = X[None,:,:]   
    nt,nc,ns = Xt.shape
    
    Xw = []
    yw = []    
    for w in gen_windows(L, ns, step=st):
        Xw.append(Xt[:,:,w])
        yw.append(int(y[w[L/2]][0]))        
    X = np.concatenate(Xw)
    y = np.array(yw)
    
    return X, y    
    