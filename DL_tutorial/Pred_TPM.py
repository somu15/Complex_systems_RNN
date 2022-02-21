#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 06:45:53 2020

@author: som
"""

# Imports
import numpy as np
np.random.seed(100)
from tensorflow import random
random.set_seed(100)

import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
os.chdir('/Users/som/Dropbox/Complex_systems_RNN/DL_tutorial')
from Kij import Kij
from scipy.stats import beta
from scipy.interpolate import interp1d
from scipy.stats import uniform
from statsmodels.distributions.empirical_distribution import ECDF

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


# Import P11 dataset EQ

dataset_path = '/Users/som/Dropbox/Complex_systems_RNN/Data/Multihazards/Eq_P11_10IM.csv'
dataset = pd.read_csv(dataset_path)
dataset.pop("IM")
dataset.pop("Time")
a1 = 1.0
b1 = 1.0
loc1 = 0
sca1 = 1

def transform1(Rec):
    return np.power((1-Rec),1/1)
    
def invtransform1(x):
    return (1-np.power(x,1))
    
dataset['Rec'] = transform1(dataset['Rec'])
dataset['P1'] = transform1(dataset['P1'])
dataset['P2'] = transform1(dataset['P2'])
dataset.tail()

train_dataset = dataset.sample(frac=1.0,random_state=100)
test_dataset = dataset.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop("Rec")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('Rec')
test_labels = test_dataset.pop('Rec')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
          model = keras.Sequential([
            layers.Dense(4, activation='softmax', input_shape=[len(train_dataset.keys())],bias_initializer='zeros'),
            layers.Dense(4, activation='softmax',bias_initializer='zeros'),
            layers.Dense(1,bias_initializer='zeros')
          ])
        
          # optimizer = tf.keras.optimizers.RMSprop(0.001)
          optimizer = tf.keras.optimizers.RMSprop(0.001)
        
          model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model

model_p11 = build_model()


EPOCHS = 500

history = model_p11.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Tim1 = np.arange(1,151,1)
# Tim2 = np.arange(1,(1000-149),1)
Tim = np.arange(1,(1000+1),1)
IMe = 1.25
# IMh = 105

dat1 = pd.DataFrame(Tim,columns=['Time'])
Dis_tim1 = Kij(IM=IMe,Haz='Eq',time=dat1['Time'])
A1,A2 = Dis_tim1.Time()
dat1['P1'] = A1.reshape(len(dat1['Time']),1)
dat1['P2'] = A2.reshape(len(dat1['Time']),1)
dat1['P1'] = transform1(dat1['P1'])
dat1['P2'] = transform1(dat1['P2'])
dat1.pop("Time")
normed_dat1 = norm(dat1)
dat_pred_p11 = invtransform1(model_p11.predict(normed_dat1).flatten())
dat_pred_p11 = 1-dat_pred_p11#[150:1000]