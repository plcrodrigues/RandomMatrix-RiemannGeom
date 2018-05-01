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

X, y = get_bci_mi_dataset(subject=3)
sel = (y == 1) | (y == 2) # get just the trials of two classes
X, y = X[sel], y[sel]
covs = Covariances().fit_transform(X)

#%% make the distance matrices

dm_riemann = make_distance_matrix(points=covs, distance=distance_riemann)
dm_euclid = make_distance_matrix(points=X, distance=distance_euclid)

distance_matrix = dm_riemann
hist, bins = np.histogram(distance_matrix.flatten(), density=True, bins=50)
tau = bins[np.argmax(hist)]

#%% make the kernel matrices 

#distance_matrix = dm_euclid
distance_matrix = dm_riemann

eps = 2*np.median(distance_matrix)**2
K = np.exp(-distance_matrix**2/eps)

#%% make the Laplacian matrix and decompose it

L = make_laplacian_matrix(kernel_matrix=K)
u,s,v = np.linalg.svd(L) 

#%% plot the spectral embedding

fig, ax = plt.subplots(facecolor='white', figsize=(7.6, 6.8))
dimx = 1; dimy = 2
ax.scatter(u[y == 1, dimx], u[y == 1, dimy], s=120, edgecolor='none', facecolor='b')
ax.scatter(u[y == 2, dimx], u[y == 2, dimy], s=120, edgecolor='none', facecolor='r')
ax.set_xticks([])
ax.set_yticks([])   


        










