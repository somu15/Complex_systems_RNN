#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:04:58 2021

@author: som
"""

import numpy as np
np.random.seed(100)
from tensorflow import random
random.set_seed(100)
import scipy.stats

import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Kij import Kij
from scipy.stats import norm
import csv

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

dataset_path = '/Users/dhulls/projects/Resilience_LDRD/Complex_systems_RNN/LSTM_based/train_EQ1.csv'
t = []
p1 = []
p2 = []
rec = []
with open(dataset_path, 'r') as file:
    reader = csv.reader(file)
    count=0
    for row in reader:
        if count>0:
            t.append(float(row[0]))
            p1.append(float(row[1]))
            p2.append(float(row[2]))
            rec.append(float(row[3]))
        count=count+1

t = np.array(t,dtype=np.float64)
p1 = np.array(p1,dtype=np.float64)
p2 = np.array(p2,dtype=np.float64)
rec = np.array(rec,dtype=np.float64)

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
    return K

def Norm2(X1,X):
    return ((X1)-np.mean((X)))/(np.std((X)))

def InvNorm2(X1,X):
    return np.exp(X1*np.std((X))+np.mean((X)))


def Norm3(X1,X):
    return ((X1)-np.mean((X)))/(np.std((X)))

def InvNorm3(X1,X):
    return (X1*np.std((X))+np.mean((X)))

def logist(X):
    return (1/(1+np.exp(-X)))

def invlogist(X):
    return (-np.log(1/X-1))

########### Implementation 1 ###########

# Ndim = 2
# k1 = np.zeros((len(p1),Ndim))
# # k1[:,0] = t
# k1[:,0] = p1 # invlogist(p1)
# k1[:,1] = p2 # invlogist(p2)
# k2 = rec
# k_tot = np.zeros((len(p1),Ndim+1))
# k_tot[:,0:Ndim] = k1
# k_tot[:,Ndim] = k2

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
# 	"""
# 	Frame a time series as a supervised learning dataset.
# 	Arguments:
# 		data: Sequence of observations as a list or NumPy array.
# 		n_in: Number of lag observations as input (X).
# 		n_out: Number of observations as output (y).
# 		dropnan: Boolean whether or not to drop rows with NaN values.
# 	Returns:
# 		Pandas DataFrame of series framed for supervised learning.
# 	"""
# 	n_vars = 1 if type(data) is list else data.shape[1]
# 	df = DataFrame(data)
# 	cols, names = list(), list()
# 	# input sequence (t-n, ... t-1)
# 	for i in range(n_in, 0, -1):
# 		cols.append(df.shift(i))
# 		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
# 	# forecast sequence (t, t+1, ... t+n)
# 	for i in range(0, n_out):
# 		cols.append(df.shift(-i))
# 		if i == 0:
# 			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
# 		else:
# 			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
# 	# put it all together
# 	agg = concat(cols, axis=1)
# 	agg.columns = names
# 	# drop rows with NaN values
# 	if dropnan:
# 		agg.dropna(inplace=True)
# 	return agg

# k_in = np.zeros((len(k_tot[699:898,:])+1,Ndim+1))
# k_in[1:len(k_tot)+1,:] = k_tot[699:898,:]
# pd1 = series_to_supervised(k_in,1,1)

# Nt = 99
# train = pd1.values[:Nt, :]
# test = pd1.values[Nt:, :]

# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
# # reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# # design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=1000, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

# # Make predictions
# yhat = model.predict(test_X)

########### Implementation 2 ###########

Ndim = 2
k1 = np.zeros((len(p1),Ndim))
# k1[:,0] = t
k1[:,0] = p1 # invlogist(p1)
k1[:,1] = p2 # invlogist(p2)
k2 = rec
k_tot = np.zeros((len(p1),Ndim+1))
k_tot[:,0:Ndim] = k1
k_tot[:,Ndim] = k2

train_ind = np.array([1,4,7,3,9])
T = 100
train_X = np.zeros((T*len(train_ind), 2*Ndim+1))
counter = 0
train_y = np.zeros((T*len(train_ind), 1))
for ii in np.arange(0,len(train_ind),1):
    start_ind = train_ind[ii]*T
    train_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind-1,0:Ndim]
    train_y[counter] = k_tot[start_ind-1,Ndim]
    for jj in np.arange(1,T,1):
        counter = counter + 1
        train_X[counter,0:Ndim+1] = k_tot[start_ind+jj-2,0:Ndim+1]
        train_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind-1+jj,0:Ndim]
        train_y[counter] = k_tot[start_ind-1+jj,Ndim]
    counter = counter + 1
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_ind = np.array([2,5,6,8])
T = 100
test_X = np.zeros((T*len(test_ind), 2*Ndim+1))
counter = 0
test_y = np.zeros((T*len(test_ind), 1))
for ii in np.arange(0,len(test_ind),1):
    start_ind = test_ind[ii]*T
    test_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind-1,0:Ndim]
    test_y[counter] = k_tot[start_ind-1,Ndim]
    for jj in np.arange(1,T,1):
        counter = counter + 1
        test_X[counter,0:Ndim+1] = k_tot[start_ind+jj-2,0:Ndim+1]
        test_X[counter,Ndim+1:2*Ndim+1] = k_tot[start_ind-1+jj,0:Ndim]
        test_y[counter] = k_tot[start_ind-1+jj,Ndim]
    counter = counter + 1
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat = np.zeros((len(test_X),1))
counter = 0
test_ind1 = np.array([0,1,2,3])
for ii in np.arange(0,len(test_X),1):
    if (ii == test_ind1*T).any():
        inp1 = test_X[ii,0,:].reshape(1,1,5)
        yhat[ii] = model.predict(inp1)
    else:
        inp1 = test_X[ii,0,:].reshape(1,1,5)
        inp1[0,0,2] = yhat[ii-1]
        yhat[ii] = model.predict(inp1)
    

