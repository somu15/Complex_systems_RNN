#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:46:03 2022

@author: dhulls
"""

import numpy as np
import scipy.stats
import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from Kij import Kij
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import rayleigh
import csv
from itertools import combinations
import random

from pandas import DataFrame
from pandas import concat
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from ML_TF import ML_TF
from pyDOE import *

def P_ds(flag = None, IM = None):

    Wind = []
    Plow = []
    Pmed = []
    Phigh = []
    File = 'Hurricane.csv'
    with open(File, 'r') as file:
        count = 0
        reader = csv.reader(file)
        for row in reader:
            if count > 0:
                Wind.append(float(row[0]))
                Plow.append(float(row[1]))
                Pmed.append(float(row[2]))
                Phigh.append(float(row[3]))
            count = count + 1
    Wind = np.array(Wind)
    Plow = np.array(Plow)
    Pmed = np.array(Pmed)
    Phigh = np.array(Phigh)
    IM = np.round(IM,2)
    index = np.where(Wind == IM)
    
    P_ex1 = Plow[index]
    P_ex2 = Pmed[index]
    P_ex3 = Phigh[index]

    if flag == 1:
        P_req = np.array([P_ex1, P_ex2, P_ex3])
    else:
        P_req = np.array([(1-P_ex1), (P_ex1-P_ex2), (P_ex2-P_ex3), (P_ex3)])

    P_req[np.isnan(P_req)] = 0

    P_req = np.abs(P_req)
    return P_req
    
def Rep_dists_Li_Ell_Hurr_Char_SMPRESS(t = None, Wind = None):

    low = 1e-3
    Times = np.array([[low, low],[5, low],[240, 120],[540, low]])
    N_states = 4;
    
    tot_time1 = 0
    tot_time2 = 0
    P_eq = P_ds(IM = Wind,flag = 0)
    
    for jj in np.arange(0,N_states,1):

        tot_time1 = tot_time1 + rayleigh(scale=0.8*Times[jj,0]).cdf(t)*P_eq[jj]
        tot_time2 = tot_time2 + rayleigh(scale=0.8*Times[jj,1]).cdf(t)*P_eq[jj]
    
    return tot_time1

# T = np.arange(0,1000,1)

Fi = np.array([0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.75,0.75,0.75,0.75])

def get_recovery_realization(N = None, F = None, Preq = None, T = None):
    total_combos = []
    for ii in np.arange(1,N,1):
        total_combos = np.append(total_combos, len(list(combinations(np.arange(1,N+1,1),ii))))
    Ft = np.zeros(N + 1)
    Ft[len(Ft)-1] = 1.0
    tt = np.zeros(N + 1)
    seq = random.sample(range(N), N)
    for ii in np.arange(1,N,1):
        ind_req = seq[ii-1] + int(np.sum(total_combos[0:ii-1]))
        Ft[ii] = F[ind_req]
        tt[ii] = tt[ii-1] + np.interp(np.random.rand(), Preq, T)
    tt[N] = tt[N-1] + np.interp(np.random.rand(), Preq, T)
    t_fin = []
    F_fin = []
    for ii in np.arange(1,len(tt),1):
        tmp = np.arange(np.floor(tt[ii-1]),np.floor(tt[ii]),1.0)
        t_fin = np.append(t_fin, tmp)
        F_fin = np.append(F_fin, np.repeat(Ft[ii-1], len(tmp)))
    tmp = np.arange(np.floor(tt[len(tt)-1]),np.floor(np.max(T)),1.0)
    t_fin = np.append(t_fin, tmp)
    F_fin = np.append(F_fin, np.repeat(Ft[len(tt)-1], len(tmp)))
    # F_fin[0] = 0.
    return T, np.interp(T, t_fin, F_fin)

def get_recovery_curve(IM = None, T = None, Sims = None, N = None):
    Freq = np.zeros(len(T))
    Preq1 = Rep_dists_Li_Ell_Hurr_Char_SMPRESS(t = T, Wind = IM)
    for ii in np.arange(0,Sims, 1):
        tt, Ft = get_recovery_realization(N = N, F = Fi, Preq = Preq1, T = T)
        Freq = Freq + Ft
    Freq = Freq / Sims
    Treq = tt
    return Treq, Freq

## Create datasets

Treq = np.arange(1,2000,20)
# Treq, Freq = get_recovery_curve(IM = 105, T = T, Sims = 1000, N = 4)
# plt.plot(Treq, Freq)
Ndim = 4
k1 = np.zeros((len(Treq),Ndim))
NTr = 7
NTe = 5
IMvec = np.concatenate((np.array([50]), uniform(50,75).rvs(NTr-2), np.array([125]), uniform(50,75).rvs(5)), axis=0) #  np.array([ 78,  80,  54,  84,  77, 116, 107,  75, 123,  52, 90, 83, 50, 110, 102]) # 
tmp = Rep_dists_Li_Ell_Hurr_Char_SMPRESS(t = Treq, Wind = IMvec[0])
k1[:,0] = tmp
k1[:,1] = tmp
k1[:,2] = tmp
k1[:,3] = tmp
dum, k2 = get_recovery_curve(IM = IMvec[0], T = Treq, Sims = 1000, N = 4)
for ii in np.arange(1,len(IMvec),1):
    tmp1 = np.zeros((len(Treq),Ndim))
    tmp = Rep_dists_Li_Ell_Hurr_Char_SMPRESS(t = Treq, Wind = IMvec[ii])
    tmp1[:,0] = tmp
    tmp1[:,1] = tmp
    tmp1[:,2] = tmp
    tmp1[:,3] = tmp
    k1 = np.append(k1,tmp1, axis = 0)
    dum, dum1 = get_recovery_curve(IM = IMvec[ii], T = Treq, Sims = 1000, N = 4)
    k2 = np.append(k2, dum1, axis = 0)
    
k_tot = np.zeros((len(Treq)*len(IMvec),Ndim+1))
k_tot[:,0:Ndim] = k1
k_tot[:,Ndim] = k2

train_ind = np.arange(0,NTr,1) # np.array([1,4,7,3,9,0,8,2])
T = 100
train_X = np.zeros((T*len(train_ind), 3)) # 3 # 2*Ndim+1  # 
counter = 0
train_y = np.zeros((T*len(train_ind), 1))
for ii in np.arange(0,len(train_ind),1):
    start_ind = train_ind[ii]*T
    # train_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind,0:Ndim]
    # train_X[counter,1:Ndim+1] = k_tot[start_ind,0:Ndim]
    
    train_X[counter,2] = k_tot[start_ind,0]
    
    
    
    train_y[counter] = k_tot[start_ind,Ndim]
    for jj in np.arange(1,T,1):
        counter = counter + 1
        # train_X[counter,0:Ndim+1] = k_tot[start_ind+jj-1,0:Ndim+1]
        # train_X[counter,0] = k_tot[start_ind+jj-1,Ndim]
        
        train_X[counter,0] = k_tot[start_ind+jj-1,0]
        train_X[counter,1] = k_tot[start_ind+jj-1,Ndim]
        
        # train_X[counter,0] = k_tot[start_ind+jj-1,0]
        
        
        
        
        # train_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind+jj,0:Ndim]
        # train_X[counter,1:Ndim+1] = k_tot[start_ind+jj,0:Ndim]
        
        train_X[counter,2] = k_tot[start_ind+jj,0]
        
        # train_X[counter,1] = k_tot[start_ind+jj,0]
        
        
        
        train_y[counter] = k_tot[start_ind+jj,Ndim]
    counter = counter + 1
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_ind = np.arange(NTr,NTr+NTe,1) # np.array([1,4,7,3,9]) # 
T = 100
test_X = np.zeros((T*len(test_ind), 3)) # 3 # 2*Ndim+1 # 
counter = 0
test_y = np.zeros((T*len(test_ind), 1))
for ii in np.arange(0,len(test_ind),1):
    start_ind = test_ind[ii]*T
    # test_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind,0:Ndim]
    # test_X[counter,1:Ndim+1] = k_tot[start_ind,0:Ndim]
    test_X[counter,2] = k_tot[start_ind,0]
    test_y[counter] = k_tot[start_ind,Ndim]
    for jj in np.arange(1,T,1):
        counter = counter + 1
        # test_X[counter,0:Ndim+1] = k_tot[start_ind+jj-1,0:Ndim+1]
        # test_X[counter,0] = k_tot[start_ind+jj-1,Ndim]
        test_X[counter,0] = k_tot[start_ind+jj-1,0]
        test_X[counter,1] = k_tot[start_ind+jj-1,Ndim]
        # test_X[counter,0] = k_tot[start_ind+jj-1,0]
        
        # test_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind+jj,0:Ndim]
        # test_X[counter,1:Ndim+1] = k_tot[start_ind+jj,0:Ndim]
        test_X[counter,2] = k_tot[start_ind+jj,0]
        # test_X[counter,1] = k_tot[start_ind+jj,0]
        
        test_y[counter] = k_tot[start_ind+jj,Ndim]
    counter = counter + 1
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

## Training method 1

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), activation="tanh")) #  , batch_size=1, stateful=True, batch_input_shape=(1, 1, 3), return_sequences = True
# model.add(LSTM(100, return_sequences = True, activation="sigmoid")) # , stateful=True, batch_input_shape=(1, 1, 3)
# model.add(LSTM(100, return_sequences = True, activation="sigmoid")) # , stateful=True, batch_input_shape=(1, 1, 3)
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=1000, batch_size=len(train_X), validation_data=(test_X, test_y), verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.yscale('log')
plt.legend()
plt.show()

# Predictions in the inputs
yhat = np.zeros(len(test_X))
counter = 0
test_ind1 = np.arange(0,NTe,1) # np.array([0,1,2,3])
for ii in np.arange(0,len(test_X),1):
    if (ii == test_ind1*T).any():
        inp1 = test_X[ii,0,:].reshape(1,1,3) # 2*Ndim+1 # 
        yhat[ii] = model.predict(inp1, batch_size=1).reshape(1)
    else:
        inp1 = test_X[ii,0,:].reshape(1,1,3) # 2*Ndim+1 # 
        # inp1[0,0,4] = yhat[ii-1]
        inp1[0,0,1] = yhat[ii-1]
        yhat[ii] = model.predict(inp1, batch_size=1).reshape(1)

# No predictions in the inputs

## Training method 2

n_batch = len(train_X)
n_epoch = 1000
n_neurons = 50
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(n_epoch):
	model.fit(train_X, train_y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()

n_batch = 1
# re-define model
new_model = Sequential()
new_model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True))
new_model.add(Dense(1))
# copy weights
old_weights = model.get_weights()
new_model.set_weights(old_weights)

# Predictions in the inputs
yhat = np.zeros(len(test_X))
counter = 0
test_ind1 = np.arange(0,NTe,1) # np.array([0,1,2,3])
for ii in np.arange(0,len(test_X),1):
    if (ii == test_ind1*T).any():
        inp1 = test_X[ii,0,:].reshape(1,1,3) # 2*Ndim+1 # 
        yhat[ii] = new_model.predict(inp1, batch_size=n_batch).reshape(1)
    else:
        inp1 = test_X[ii,0,:].reshape(1,1,3) # 2*Ndim+1 # 
        # inp1[0,0,4] = yhat[ii-1]
        inp1[0,0,1] = yhat[ii-1]
        yhat[ii] = new_model.predict(inp1, batch_size=n_batch).reshape(1)

# No predictions in the inputs
yhat = np.zeros(len(test_X))
counter = 0
test_ind1 = np.arange(0,NTe,1) # np.array([0,1,2,3])
for ii in np.arange(0,len(test_X),1):
    inp1 = test_X[ii,0,:].reshape(1,1,2) # 3 # 2*Ndim+1 # 
    yhat[ii] = new_model.predict(inp1, batch_size=n_batch).reshape(1)



