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

dataset_path = '/Users/som/Dropbox/Complex_systems_RNN/Kriging_based/train_EQ1.csv'
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

def Norm1(X):
    return X/1000

def InvNorm1(X):
    return X*1000

# dataset = pd.read_csv(dataset_path)
# dataset.pop("IM")
# dataset.pop("Time")

# a1 = 1.0
# b1 = 1.0
# loc1 = 0
# sca1 = 1

# def transform(Rec):
#     # return -np.log(1/Rec-1)
#     return norm.ppf(Rec)
    

# def invtransform(x):
#     # return 1/(1+np.exp(-x))
#     return norm.cdf(x)

# dataset = dataset.astype('float32')

# dataset['Rec'] = transform(dataset['Rec'])

# dataset['P1'] = transform(dataset['P1'])

# dataset['P2'] = transform(dataset['P2'])

# dataset.tail()

# # Generate a KDE from the empirical sample
# sample_pdf = scipy.stats.gaussian_kde(np.array(dataset['Rec']))

# # Sample new datapoints from the KDE
# new_sample_data = sample_pdf.resample(10000).T[:,0]

# Split the data into train and test

# train_dataset = dataset.sample(frac=1.0,random_state=100)
# test_dataset = dataset.drop(train_dataset.index)

# # Inspect the data

# # sns.pairplot(train_dataset[["Rec", "P1", "P2", "Time"]], diag_kind="kde")
# # sns.pairplot(train_dataset[["Rec", "IM"]], diag_kind="kde")

# train_stats = train_dataset.describe()
# train_stats.pop("Rec")
# train_stats = train_stats.transpose()
# train_stats

# # Split features from labels

# train_labels = train_dataset.pop('Rec')
# test_labels = test_dataset.pop('Rec')

# # Normalize the data

# def norm1(x):
#   # return (x - train_stats['mean']) / train_stats['std']
#   return x
# normed_train_data = norm1(train_dataset)
# normed_test_data = norm1(test_dataset)

# k1 = np.array(normed_train_data, dtype=np.float64)
# k2 = np.array(train_labels, dtype=np.float64)
# indreq = np.where(train_labels!=np.inf)
# k2 = k2[indreq]
# k1 = k1[indreq,:]
# indreq = np.where(train_labels==-np.inf)
# k2[indreq] = -12

k1 = np.zeros((len(p1),3))
k1[:,0] = t
k1[:,1] = p1
k1[:,2] = p2
ML = ML_TF(obs_ind = (k1), obs = (rec))
amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)

# dataset_path1 = '/Users/som/Dropbox/Complex_systems_RNN/Kriging_based/test_EQ.csv'
# data1 = pd.read_csv(dataset_path1)
# # data1["IM"] = np.log(data1["IM"])
# Time1 = data1.pop("Time")
# # data1.pop("P1")
# # data1.pop("P2")
# data1['Rec'] = transform(data1['Rec'])
# data1['P1'] = transform(data1['P1'])
# data1['P2'] = transform(data1['P2'])
# data1_labels = data1.pop('Rec')
# normed_data1 = norm1(data1)
# samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = normed_data1, num_samples=500)
# data1_pred = np.mean(np.array(samples1),axis=0)

# plt.figure(2)
# plt.plot(Time1,invtransform(data1_labels),label='Exact')
# plt.plot(Time1,invtransform(data1_pred).reshape(1997),label='GP')
# plt.xlabel('Time [days]')
# plt.ylabel('Functionality')
# plt.xlim([0, 300])
# plt.ylim([0, 1])
# plt.legend()

dataset_path1 = '/Users/som/Dropbox/Complex_systems_RNN/Kriging_based/test_EQ_0_7_1.csv'
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
k1_t = np.zeros((len(p1_t),3))
k1_t[:,0] = t_t
k1_t[:,1] = p1_t
k1_t[:,2] = p2_t
samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = (k1_t), num_samples=500)
data1_pred = (np.mean(np.array(samples1),axis=0))