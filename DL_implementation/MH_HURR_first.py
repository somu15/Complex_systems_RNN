#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 06:44:32 2020

@author: som
"""

import numpy as np
np.random.seed(100)
from tensorflow import random
random.set_seed(100)
from IPython.display import display
import pickle

import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
os.chdir('/Users/som/Dropbox/Complex_systems_RNN/DL_implementation')
from Kij import Kij
from scipy.stats import beta
from scipy.interpolate import interp1d
from scipy.stats import uniform
from scipy.stats import expon
from statsmodels.distributions.empirical_distribution import ECDF

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

## Predict P11

dataset_path1 = '/Users/som/Dropbox/Complex_systems_RNN/Data/Multihazards/Hurr_P11_7IM.csv'
dataset1 = pd.read_csv(dataset_path1)
dataset1.pop("IM")
dataset1.pop("Time")
a1 = 1.0
b1 = 1.0
loc1 = 0
sca1 = 1

def transform1(Rec):
    return np.power((1-Rec),1/4)
    
def invtransform1(x):
    return (1-np.power(x,4))

def transform2(Rec):
    return np.power((1-Rec),1/4)
    
def invtransform2(x):
    return (1-np.power(x,4))
    
dataset1['Rec'] = transform1(dataset1['Rec'])
dataset1['P1'] = transform1(dataset1['P1'])
dataset1['P2'] = transform1(dataset1['P2'])
dataset1.tail()

train_dataset1 = dataset1.sample(frac=1.0,random_state=100)
test_dataset1 = dataset1.drop(train_dataset1.index)
train_stats1 = train_dataset1.describe()
train_stats1.pop("Rec")
train_stats1 = train_stats1.transpose()
train_stats1

train_labels1 = train_dataset1.pop('Rec')
test_labels1 = test_dataset1.pop('Rec')

def norm1(x):
  return (x - train_stats1['mean']) / train_stats1['std']
normed_train_data1 = norm1(train_dataset1)
normed_test_data1 = norm1(test_dataset1)

def build_model1():
          model1 = keras.Sequential([
            layers.Dense(6, activation='softmax', input_shape=[len(train_dataset1.keys())],bias_initializer='zeros'),
            # layers.Dense(4, activation='softmax',bias_initializer='zeros'),
            layers.Dense(1,bias_initializer='zeros')
          ])
        
          # optimizer = tf.keras.optimizers.RMSprop(0.001)
          optimizer = tf.keras.optimizers.RMSprop(0.001)
        
          model1.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model1

model_p11 = build_model1()


EPOCHS = 1000

history1 = model_p11.fit(
  normed_train_data1, train_labels1,
  epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)

hist1 = pd.DataFrame(history1.history)
hist1['epoch'] = history1.epoch
hist1.tail()

## Predict P22

dataset_path2 = '/Users/som/Dropbox/Complex_systems_RNN/Data/Multihazards/Hurr_P22_7IM.csv'
dataset2 = pd.read_csv(dataset_path2)
dataset2.pop("IM")
dataset2.pop("Time")
a1 = 1.0
b1 = 1.0
loc1 = 0
sca1 = 1
    
dataset2['Rec'] = transform1(dataset2['Rec'])
dataset2['P1'] = transform1(dataset2['P1'])
dataset2['P2'] = transform1(dataset2['P2'])
dataset2.tail()

train_dataset2 = dataset2.sample(frac=1.0,random_state=100)
test_dataset2 = dataset2.drop(train_dataset2.index)
train_stats2 = train_dataset2.describe()
train_stats2.pop("Rec")
train_stats2 = train_stats2.transpose()
train_stats2

train_labels2 = train_dataset2.pop('Rec')
test_labels2 = test_dataset2.pop('Rec')

def norm2(x):
  return (x - train_stats2['mean']) / train_stats2['std']
normed_train_data2 = norm2(train_dataset2)
normed_test_data2 = norm2(test_dataset2)

def build_model2():
          model1 = keras.Sequential([
            layers.Dense(6, activation='softmax', input_shape=[len(train_dataset2.keys())],bias_initializer='zeros'),
            # layers.Dense(4, activation='softmax',bias_initializer='zeros'),
            layers.Dense(1,bias_initializer='zeros')
          ])
        
          # optimizer = tf.keras.optimizers.RMSprop(0.001)
          optimizer = tf.keras.optimizers.RMSprop(0.001)
        
          model1.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model1

model_p22 = build_model2()


EPOCHS = 1000

history2 = model_p22.fit(
  normed_train_data2, train_labels2,
  epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)

hist2 = pd.DataFrame(history2.history)
hist2['epoch'] = history2.epoch
hist2.tail()

## Predict SH

dataset_path3 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for improved disf match/Hurr_10IM.csv'
dataset3 = pd.read_csv(dataset_path3)
dataset3.pop("IM")
dataset3.pop("Time")
a1 = 1.0
b1 = 1.0
loc1 = 0
sca1 = 1
    
dataset3['Rec'] = transform1(dataset3['Rec'])
dataset3['P1'] = transform1(dataset3['P1'])
dataset3['P2'] = transform1(dataset3['P2'])
dataset3.tail()

train_dataset3 = dataset3.sample(frac=1.0,random_state=100)
test_dataset3 = dataset3.drop(train_dataset3.index)
train_stats3 = train_dataset3.describe()
train_stats3.pop("Rec")
train_stats3 = train_stats3.transpose()
train_stats3

train_labels3 = train_dataset3.pop('Rec')
test_labels3 = test_dataset3.pop('Rec')

def norm3(x):
  return (x - train_stats3['mean']) / train_stats3['std']
normed_train_data3 = norm3(train_dataset3)
normed_test_data3 = norm3(test_dataset3)

def build_model3():
          model1 = keras.Sequential([
            layers.Dense(6, activation='softmax', input_shape=[len(train_dataset3.keys())],bias_initializer='zeros'),
            # layers.Dense(4, activation='softmax',bias_initializer='zeros'),
            layers.Dense(1,bias_initializer='zeros')
          ])
        
          # optimizer = tf.keras.optimizers.RMSprop(0.001)
          optimizer = tf.keras.optimizers.RMSprop(0.001)
        
          model1.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model1

model_SH = build_model3()


EPOCHS = 1000

history3 = model_SH.fit(
  normed_train_data3, train_labels3,
  epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)

hist3 = pd.DataFrame(history3.history)
hist3['epoch'] = history3.epoch
hist3.tail()

## Predict MH

dataset_path4 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for improved disf match/MH_HE_10IM.csv'
dataset4 = pd.read_csv(dataset_path4)
dataset4.pop("IMe")
dataset4.pop("IMh")
dataset4.pop("Tint")
dataset4.pop("Time")
a1 = 1.0
b1 = 1.0
loc1 = 0
sca1 = 1
    
dataset4['Rec'] = transform1(dataset4['Rec'])
dataset4['P1'] = transform1(dataset4['P1'])
dataset4['P2'] = transform1(dataset4['P2'])
dataset4.tail()

train_dataset4 = dataset4.sample(frac=1.0,random_state=100)
test_dataset4 = dataset4.drop(train_dataset4.index)
train_stats4 = train_dataset4.describe()
train_stats4.pop("Rec")
train_stats4 = train_stats4.transpose()
train_stats4

train_labels4 = train_dataset4.pop('Rec')
test_labels4 = test_dataset4.pop('Rec')

def norm4(x):
  return (x - train_stats4['mean']) / train_stats4['std']
normed_train_data4 = norm4(train_dataset4)
normed_test_data4 = norm4(test_dataset4)

def build_model4():
          model1 = keras.Sequential([
            layers.Dense(6, activation='softmax', input_shape=[len(train_dataset4.keys())],bias_initializer='zeros'),
            # layers.Dense(4, activation='softmax',bias_initializer='zeros'),
            layers.Dense(1,bias_initializer='zeros')
          ])
        
          # optimizer = tf.keras.optimizers.RMSprop(0.001)
          optimizer = tf.keras.optimizers.RMSprop(0.001)
        
          model1.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model1

model_MH = build_model4()


EPOCHS = 1000

history4 = model_MH.fit(
  normed_train_data4, train_labels4,
  epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)

hist4 = pd.DataFrame(history4.history)
hist4['epoch'] = history4.epoch
hist4.tail()

## Predict single recovery curve

IMe = [0.75]
IMh = [65]
T_int = 150
N_int = 1
N_sims = 10000

fin_rec = np.zeros((2000,len(IMe),len(IMh)))
count = 0

for ii in np.arange(0,len(IMe),1):
    for jj in np.arange(0,len(IMh),1):
        IMe_req = IMe[ii]
        IMh_req = IMh[jj]
        rec_avg_tint = np.zeros((1,2000))
        for kk in np.arange(0,N_int,1):
            # T_int = rv_tint.rvs()+1
            if T_int>2000:T_int=1900.0
            
            Tim1 = np.arange(1,(int(np.round(T_int))+1),1)
            Tim2 = np.arange(1,(2000-(int(np.round(T_int))-1)),1)
            Tim = np.arange(1,(2000+1),1)
            Tim3 = np.arange(1,(2000+1),1)
            
            dat = pd.DataFrame(Tim1,columns=['Time'])
            Dis_tim = Kij(IM=IMh_req,Haz='Hurr',time=dat['Time'])
            A1,A2 = Dis_tim.Time()
            dat['P1'] = A1.reshape(len(dat['Time']),1)
            dat['P2'] = A2.reshape(len(dat['Time']),1)
            dat['P1'] = transform2(dat['P1'])
            dat['P2'] = transform2(dat['P2'])
            dat.pop("Time")
            normed_dat = norm3(dat)
            dat_pred1 = invtransform2(model_SH.predict(normed_dat).flatten())
            
            dat1 = pd.DataFrame(Tim,columns=['Time'])
            Dis_tim1 = Kij(IM=IMh_req,Haz='Hurr',time=dat1['Time'])
            A1,A2 = Dis_tim1.Time()
            dat1['P1'] = A1.reshape(len(dat1['Time']),1)
            dat1['P2'] = A2.reshape(len(dat1['Time']),1)
            dat1['P1'] = transform1(dat1['P1'])
            dat1['P2'] = transform1(dat1['P2'])
            dat1.pop("Time")
            normed_dat1 = norm1(dat1)
            dat_pred_p11 = invtransform1(model_p11.predict(normed_dat1).flatten())
            dat_pred_p11 = 1-dat_pred_p11[int(np.round(T_int)):2000]
            dat_pred_p11[np.where(dat_pred_p11>1)]=1
            dat_pred_p11[0] = 0
            
            dat2 = pd.DataFrame(Tim,columns=['Time'])
            Dis_tim2 = Kij(IM=IMh_req,Haz='Hurr',time=dat2['Time'])
            A1,A2 = Dis_tim2.Time()
            dat2['P1'] = A1.reshape(len(dat2['Time']),1)
            dat2['P2'] = A2.reshape(len(dat2['Time']),1)
            dat2['P1'] = transform1(dat2['P1'])
            dat2['P2'] = transform1(dat2['P2'])
            dat2.pop("Time")
            normed_dat2 = norm2(dat2)
            dat_pred_p22 = invtransform1(model_p22.predict(normed_dat2).flatten())
            dat_pred_p22 = 1-dat_pred_p22[int(np.round(T_int)):2000]
            dat_pred_p22[np.where(dat_pred_p22>1)]=1
            dat_pred_p22[0] = 0
            
            Dis_tim3 = Kij(IM=IMe_req,Haz='Eq',time=Tim3)
            A1,A2 = Dis_tim3.Time()
            A1[0] = 0
            A2[0] = 0
            A1[len(A1)-1] = 1
            A2[len(A2)-1] = 1
            
            rv1 = uniform()
            p1,p2 = np.unique(dat_pred_p11,return_index=True)
            p11_f = interp1d(p1,Tim2[p2],kind='linear')
            p11_times = p11_f(rv1.rvs(N_sims))
            rv2 = uniform()
            p1,p2 = np.unique(A1,return_index=True)
            k12_f = interp1d(p1,Tim3[p2])
            k12_times = k12_f(rv2.rvs(N_sims))
            ecdf_k12 = ECDF(p11_times+k12_times) # p11_times+
            
            rv3 = uniform()
            p1,p2 = np.unique(dat_pred_p22,return_index=True)
            p1[np.where(p1==np.max(p1))] = 1
            p22_f = interp1d(p1,Tim2[p2])
            p22_times = p22_f(rv3.rvs(N_sims))
            rv4 = uniform()
            p1,p2 = np.unique(A2,return_index=True)
            k23_f = interp1d(p1,Tim3[p2])
            k23_times = k23_f(rv4.rvs(N_sims))
            ecdf_k23 = ECDF(p22_times+k23_times) # p22_times+
            
            dat_MH = pd.DataFrame(Tim2,columns=['Time'])
            K = ecdf_k12(Tim2)
            dat_MH['P1'] = K.reshape(len(Tim2),1)
            K = ecdf_k23(Tim2)
            dat_MH['P2'] = K.reshape(len(Tim2),1)
            dat_MH['P1'] = transform2(dat_MH['P1'])
            dat_MH['P2'] = transform2(dat_MH['P2'])
            dat_MH.pop("Time")
            normed_dat_MH = norm4(dat_MH)
            dat_pred_MH = invtransform2(model_MH.predict(normed_dat_MH).flatten())
            
            rec_avg_tint = rec_avg_tint +  np.concatenate((dat_pred1, dat_pred_MH), axis=0)
            
        rec_avg_tint = rec_avg_tint/N_int
        fin_rec[:,ii,jj] = rec_avg_tint[0]
        count = count + 1
        display(count/(len(IMe)*len(IMh)))
        
K = rec_avg_tint[0]

## Predict multiple recovery curves

IMe = [0.75]
IMh = [65]
T_int = 150
N_int = 1
N_sims = 10000

fin_rec = np.zeros((2000,len(IMe),len(IMh)))
count = 0

for ii in np.arange(0,len(IMe),1):
    for jj in np.arange(0,len(IMh),1):
        IMe_req = IMe[ii]
        IMh_req = IMh[jj]
        rec_avg_tint = np.zeros((1,2000))
        for kk in np.arange(0,N_int,1):
            # T_int = rv_tint.rvs()+1
            if T_int>2000:T_int=1900.0
            
            Tim1 = np.arange(1,(int(np.round(T_int))+1),1)
            Tim2 = np.arange(1,(2000-(int(np.round(T_int))-1)),1)
            Tim = np.arange(1,(2000+1),1)
            Tim3 = np.arange(1,(2000+1),1)
            
            dat = pd.DataFrame(Tim1,columns=['Time'])
            Dis_tim = Kij(IM=IMh_req,Haz='Hurr',time=dat['Time'])
            A1,A2 = Dis_tim.Time()
            dat['P1'] = A1.reshape(len(dat['Time']),1)
            dat['P2'] = A2.reshape(len(dat['Time']),1)
            dat['P1'] = transform2(dat['P1'])
            dat['P2'] = transform2(dat['P2'])
            dat.pop("Time")
            normed_dat = norm3(dat)
            dat_pred1 = invtransform2(model_SH.predict(normed_dat).flatten())
            
            dat1 = pd.DataFrame(Tim,columns=['Time'])
            Dis_tim1 = Kij(IM=IMh_req,Haz='Hurr',time=dat1['Time'])
            A1,A2 = Dis_tim1.Time()
            dat1['P1'] = A1.reshape(len(dat1['Time']),1)
            dat1['P2'] = A2.reshape(len(dat1['Time']),1)
            dat1['P1'] = transform1(dat1['P1'])
            dat1['P2'] = transform1(dat1['P2'])
            dat1.pop("Time")
            normed_dat1 = norm1(dat1)
            dat_pred_p11 = invtransform1(model_p11.predict(normed_dat1).flatten())
            dat_pred_p11 = 1-dat_pred_p11[int(np.round(T_int)):2000]
            dat_pred_p11[np.where(dat_pred_p11>1)]=1
            dat_pred_p11[0] = 0
            
            dat2 = pd.DataFrame(Tim,columns=['Time'])
            Dis_tim2 = Kij(IM=IMh_req,Haz='Hurr',time=dat2['Time'])
            A1,A2 = Dis_tim2.Time()
            dat2['P1'] = A1.reshape(len(dat2['Time']),1)
            dat2['P2'] = A2.reshape(len(dat2['Time']),1)
            dat2['P1'] = transform1(dat2['P1'])
            dat2['P2'] = transform1(dat2['P2'])
            dat2.pop("Time")
            normed_dat2 = norm2(dat2)
            dat_pred_p22 = invtransform1(model_p22.predict(normed_dat2).flatten())
            dat_pred_p22 = 1-dat_pred_p22[int(np.round(T_int)):2000]
            dat_pred_p22[np.where(dat_pred_p22>1)]=1
            dat_pred_p22[0] = 0
            
            Dis_tim3 = Kij(IM=IMe_req,Haz='Eq',time=Tim3)
            A1,A2 = Dis_tim3.Time()
            A1[0] = 0
            A2[0] = 0
            A1[len(A1)-1] = 1
            A2[len(A2)-1] = 1
            
            rv1 = uniform()
            p1,p2 = np.unique(dat_pred_p11,return_index=True)
            p11_f = interp1d(p1,Tim2[p2],kind='linear')
            p11_times = p11_f(rv1.rvs(N_sims))
            rv2 = uniform()
            p1,p2 = np.unique(A1,return_index=True)
            k12_f = interp1d(p1,Tim3[p2])
            k12_times = k12_f(rv2.rvs(N_sims))
            ecdf_k12 = ECDF(p11_times+k12_times) # p11_times+
            
            rv3 = uniform()
            p1,p2 = np.unique(dat_pred_p22,return_index=True)
            p1[np.where(p1==np.max(p1))] = 1
            p22_f = interp1d(p1,Tim2[p2])
            p22_times = p22_f(rv3.rvs(N_sims))
            rv4 = uniform()
            p1,p2 = np.unique(A2,return_index=True)
            k23_f = interp1d(p1,Tim3[p2])
            k23_times = k23_f(rv4.rvs(N_sims))
            ecdf_k23 = ECDF(p22_times+k23_times) # p22_times+
            
            dat_MH = pd.DataFrame(Tim2,columns=['Time'])
            K = ecdf_k12(Tim2)
            dat_MH['P1'] = K.reshape(len(Tim2),1)
            K = ecdf_k23(Tim2)
            dat_MH['P2'] = K.reshape(len(Tim2),1)
            dat_MH['P1'] = transform2(dat_MH['P1'])
            dat_MH['P2'] = transform2(dat_MH['P2'])
            dat_MH.pop("Time")
            normed_dat_MH = norm4(dat_MH)
            dat_pred_MH = invtransform2(model_MH.predict(normed_dat_MH).flatten())
            
            rec_avg_tint = rec_avg_tint +  np.concatenate((dat_pred1, dat_pred_MH), axis=0)
            
        rec_avg_tint = rec_avg_tint/N_int
        fin_rec[:,ii,jj] = rec_avg_tint[0]
        count = count + 1
        display(count/(len(IMe)*len(IMh)))
        
K = rec_avg_tint[0]