#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:48:54 2018

@author: plcr
"""

import numpy as np

def make_distance_matrix(points, distance):
    Npoints = points.shape[0]
    distmatrix = np.zeros((Npoints, Npoints))
    for ii,pi in enumerate(points):
        for jj,pj in enumerate(points):
            distmatrix[ii,jj] = distance(pi,pj)            
    return distmatrix

def make_laplacian_matrix(kernel_matrix):   
    n = len(kernel_matrix)
    K = kernel_matrix
    d = np.sqrt(np.dot(K, np.ones(len(K))))
    L = n * np.divide(K, np.outer(d, d)) # laplacian matrix in Romain's paper    
    return L