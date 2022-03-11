#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:00:47 2022

@author: shumonkoga
"""

import numpy as np
import math
from numpy.linalg import inv, det, slogdet
from utils import unicycle_dyn, circle_SDF, Gaussian_CDF, diff_FoV_land

## Time sequence
t_fin = 5 #Total time
dt = 1 #Time step size
tf =  math.floor(t_fin / dt) #Number of time sequence
time = np.arange(0, tf) #Time sequence

##State and action
x = np.zeros((3,tf+1))
# u = np.zeros((3,tf))
u = np.array([np.pi/2*(np.arange(5)/5-0.5), 0.5*(np.arange(5)/5-0.5), np.zeros(5)], dtype=np.float32)
print(u)
u *= 2

##Landmark map
n_land = 2
y = np.zeros((2 * n_land,1))
y[:2,0], y[2:4,0] = np.array([1, 1]), np.array([-1, -1])
Y = np.diag(np.array([1, 1, 2, 2]))
r = 2 #radius of circle FoV
std = .5 #sensor noise standard deviation
kappa = .2 #smoothness of diff FoV

for t in range(tf):
    print(x[:, t])
    # u[:2,t] = np.random.uniform(low=-2, high=2, size=2) #Action
    # u[2,t] = 0 #Angular input is set to zero for simple first trial
    x[:,t+1] = unicycle_dyn(x[:,t], u[:,t], dt) #Environment 
    V_jj_inv = diff_FoV_land(x[:,t+1],y,n_land,r,kappa,std) #diff_FoV
    Y_next = Y + V_jj_inv #Info_update
    r_step = slogdet(Y_next)[1] - slogdet(Y)[1] #per-step reward
    Y = Y_next
    print( "reward = ", r_step)
