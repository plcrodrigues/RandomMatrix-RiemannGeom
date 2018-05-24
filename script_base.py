#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""  
Created on Tue May  1 09:42:42 2018

@author: plcr
"""

import numpy as np
import matplotlib.pyplot as plt

from pyriemann.utils.distance import distance_riemann, distance_euclid
from pyriemann.estimation import Covariances

from utilities.data_handler import get_bci_mi_dataset, get_alpha_waves_dataset
from utilities.data_process import make_distance_matrix, make_laplacian_matrix

#%% get the datasets

subject = 1
X, y = get_alpha_waves_dataset(subject, session=1)
idx = np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0]]) 
X, y = X[idx], y[idx]
covs = Covariances().fit_transform(X)

Xn = np.stack([Xi/np.linalg.norm(Xi) for Xi in X])
yn = y

#%% make the distance matrices

dm_riemann = make_distance_matrix(points=covs, distance=distance_riemann)
dm_euclid = make_distance_matrix(points=X, distance=distance_euclid)

#%%

distance_matrix = dm_riemann
eps = 2*np.median(distance_matrix)**2
K = np.exp(-distance_matrix**2/eps)
L = make_laplacian_matrix(kernel_matrix=K)
u,s,v = np.linalg.svd(L) 

#%%

   










