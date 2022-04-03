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

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from ML_TF import ML_TF
from pyDOE import *

dataset_path = '/Users/dhulls/projects/Resilience_LDRD/Complex_systems_RNN/Kriging_based/train_EQ1.csv'
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

# def InvNorm1(X1,X):
#     return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

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

# k1 = np.zeros((len(p1),3))
# k1[:,0] = t
# k1[:,1] = p1 # invlogist(p1)
# k1[:,2] = p2 # invlogist(p2)
Ndim = 2
k1 = np.zeros((len(p1),Ndim))
# k1[:,0] = t
k1[:,0] = p1 # invlogist(p1)
k1[:,1] = p2 # invlogist(p2)
k2 = rec
# k2[np.where(k2==1.)] = 0.9999
# k2[np.where(k2==0.)] = 0.0001
# k2 = invlogist(k2)

######### Gaussian Process #############

# ML = ML_TF(obs_ind = Norm1(k1,k1,Ndim), obs = Norm3(k2,k2))
# amp1, len1 = ML.GP_train_kernel(num_iters = 1000, amp_init = 1., len_init = np.array([1.,1.,1.]))

# dataset_path1 = '/Users/dhulls/projects/Resilience_LDRD/Complex_systems_RNN/Kriging_based/test_EQ_0_7_1.csv'
# t_t = []
# p1_t = []
# p2_t = []
# rec_t = []
# with open(dataset_path1, 'r') as file:
#     reader = csv.reader(file)
#     count=0
#     for row in reader:
#         if count>0:
#             t_t.append(float(row[0]))
#             p1_t.append(float(row[1]))
#             p2_t.append(float(row[2]))
#             rec_t.append(float(row[3]))
#         count=count+1

# t_t = np.array(t_t,dtype=np.float64)
# p1_t = np.array(p1_t,dtype=np.float64)
# p2_t = np.array(p2_t,dtype=np.float64)
# rec_t = np.array(rec_t,dtype=np.float64)
# k1_t = np.zeros((len(p1_t),Ndim))
# k1_t[:,0] = t_t
# k1_t[:,1] = p1_t
# k1_t[:,2] = p2_t
# samples1 = ML.GP_predict_kernel(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(k1_t,k1,Ndim), num_samples=500)
# data1_pred = InvNorm3(np.mean(np.array(samples1),axis=0),rec)
# data1_std = np.std(InvNorm3(np.array(samples1),rec),axis=0)

############# Deep Neural Network #############

ML = ML_TF(obs_ind = k1, obs = k2)
DNN_model = ML.DNN_train(dim=Ndim, seed=100, neurons1=20, neurons2=20, learning_rate=0.002, epochs=5000)

dataset_path1 = '/Users/dhulls/projects/Resilience_LDRD/Complex_systems_RNN/Kriging_based/test_EQ_0_7_1.csv'
t_t = []
p1_t = []
p2_t = []
rec_t = []
with open(dataset_path1, 'r') as file:
    reader = csv.reader(file)
    count=0
    for row in reader:
        if count>0:
            t_t.append(float(row[0]))
            p1_t.append(float(row[1]))
            p2_t.append(float(row[2]))
            rec_t.append(float(row[3]))
        count=count+1

t_t = np.array(t_t,dtype=np.float64)
p1_t = np.array(p1_t,dtype=np.float64)
p2_t = np.array(p2_t,dtype=np.float64)
rec_t = np.array(rec_t,dtype=np.float64)
k1_t = np.zeros((len(p1_t),Ndim))
# k1_t[:,0] = t_t # 
k1_t[:,0] = p1_t
k1_t[:,1] = p2_t
DNN_pred1 = ML.DNN_pred(k1,k2,DNN_model,Ndim,k1_t)
