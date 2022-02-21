#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 06:11:29 2020

@author: som
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 06:14:34 2020

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import dataset

dataset_path = '/Users/som/Dropbox/Complex_systems_RNN/Data/DL_test_data_Eq_reverse.csv'
# column_names = ['Time','P1','P2','IM','Rec']

# dataset = pd.read_csv(dataset_path, names=column_names,
#                       na_values = "?", comment='\t',
#                       sep=" ", skipinitialspace=True)

dataset = pd.read_csv(dataset_path)
# dataset.pop("IM")
# dataset.pop("Time")
# dataset["IM"] = np.log(dataset["IM"])
# dataset.pop("P1")
# dataset.pop("P2")
a1 = 1.0
b1 = 1.0
loc1 = 0
sca1 = 1

def transform(Rec):
    # Rec = beta.ppf(Rec,a1,b1,loc1,sca1)
    # Fin = np.zeros((len(Rec),1))
    # for ii in np.arange(0,len(Rec),1):
    #     if ((1-Rec[ii])<0.04):
    #         Fin[ii] = np.power((1-Rec[ii]),1/4)
    #     else:
    #         Fin[ii] = 1-Rec[ii]
    # return Fin
    return np.power((1-Rec),1/4)
    # return (1/(1+np.exp(-(1-Rec))))

def invtransform(x):
    # Fin = np.zeros(len(x))
    # for ii in np.arange(0,len(x),1):
    #     if ((1-np.power(x[ii],4))<0.04):
    #         Fin[ii] = (1-np.power(x[ii],4))
    #     else:
    #         Fin[ii] = 1-x[ii]
    # return Fin # (1-np.power(x,4))
    return (1-np.power(x,4))
    # return (1+np.log(1/(x)-1))
    #beta.cdf(x,a1,b1,loc1,sca1)

dataset['Rec'] = transform(dataset['Rec'])

dataset['P1'] = transform(dataset['P1'])

dataset['P2'] = transform(dataset['P2'])

dataset.tail()

# Split the data into train and test

train_dataset = dataset.sample(frac=1.0,random_state=100)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data

# sns.pairplot(train_dataset[["Rec", "P1", "P2", "Time"]], diag_kind="kde")
# sns.pairplot(train_dataset[["Rec", "IM"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Rec")
train_stats = train_stats.transpose()
train_stats

# Split features from labels

train_labels = train_dataset.pop('Rec')
test_labels = test_dataset.pop('Rec')

# Normalize the data

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Build the model ,kernel_regularizer='l2'

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

model = build_model()

# Inspect the model

model.summary()

# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# example_result

# Train the model

EPOCHS = 3000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
# plotter.plot({'Basic': history}, metric = "mae")
# # plt.ylim([0, 25])
# plt.ylabel('MAE [Rec]')

# plotter.plot({'Basic': history}, metric = "mse")
# # plt.ylim([0, 25])
# plt.ylabel('MSE [Rec]')

# Callback

# model = build_model()

# The patience parameter is the amount of epochs to check for improvement
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# early_history = model.fit(normed_train_data, train_labels, 
#                     epochs=EPOCHS, validation_split = 0.2, verbose=0, 
#                     callbacks=[early_stop, tfdocs.modeling.EpochDots()])
# plotter.plot({'Early Stopping': early_history}, metric = "mae")
# plt.ylim([0, 10])
# plt.ylabel('MAE [Rec]')
# loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
# print("Testing set Mean Abs Error: {:5.2f} Rec".format(mae))



# Make predictions

# test_predictions = model.predict(normed_test_data).flatten()

# CORR = np.corrcoef(invtransform(test_labels), invtransform(test_predictions))

# a = plt.axes(aspect='equal')
# plt.scatter(invtransform(test_labels), invtransform(test_predictions))
# plt.xlabel('True Values [Rec]')
# plt.ylabel('Predictions [Rec]')
# lims = [0, 1]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)

# error = invtransform(test_predictions) - invtransform(test_labels)
# STD = np.std(error)
# plt.hist(error, bins = 25)
# plt.xlabel("Prediction Error [Rec]")
# _ = plt.ylabel("Count")

## Verify model

# dataset_path1 = '/Users/som/Dropbox/Complex_systems_RNN/Data/DL_verify_EQ_1_3.csv'

# data1 = pd.read_csv(dataset_path1)

# Rec = data1['Rec']
# r1 = Rec == 0.0
# r2 = Rec == 1.0
# Rec[r1] = 0.001
# Rec[r2] = 0.999
# Rec = -np.log(1/Rec - 1)
# data1['Rec'] = Rec

# Rec = data1['P1']
# r1 = Rec == 0.0
# r2 = Rec == 1.0
# Rec[r1] = 0.001
# Rec[r2] = 0.999
# Rec = -np.log(1/Rec - 1)
# data1['P1'] = Rec

# Rec = data1['P2']
# r1 = Rec == 0.0
# r2 = Rec == 1.0
# Rec[r1] = 0.001
# Rec[r2] = 0.999
# Rec = -np.log(1/Rec - 1)
# data1['P2'] = Rec

# data1.tail()

# data1_labels = data1.pop('Rec')

# normed_data1 = norm(data1)

# data1_pred = model.predict(normed_data1).flatten()

# a = plt.axes(aspect='equal')
# plt.scatter(sigm(data1_labels), sigm(data1_pred))
# plt.xlabel('True Values [Rec]')
# plt.ylabel('Predictions [Rec]')
# lims = [0, 1]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)

# plt.plot(data1['Time'],sigm(data1_labels),label='Exact')
# plt.plot(data1['Time'],sigm(data1_pred),label='Prediction')
# plt.xlabel('Time [days]')
# plt.ylabel('Functionality')
# plt.xlim([0, 2000])
# plt.ylim([0, 1])
# plt.legend()

## Verify model for multiple recovery curves

dataset_path1 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for sequence/DL_verify_EQ_0_7.csv'
data1 = pd.read_csv(dataset_path1)
# data1["IM"] = np.log(data1["IM"])
Time1 = data1.pop("Time")
# data1.pop("P1")
# data1.pop("P2")
data1['Rec'] = transform(data1['Rec'])
data1['P1'] = transform(data1['P1'])
data1['P2'] = transform(data1['P2'])
data1_labels = data1.pop('Rec')
normed_data1 = norm(data1)
data1_pred = model.predict(normed_data1).flatten()


dataset_path2 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for sequence/DL_verify_EQ_0_9.csv'
data2 = pd.read_csv(dataset_path2)
# data2["IM"] = np.log(data2["IM"])
data2.pop("Time")
# data2.pop("P1")
# data2.pop("P2")
data2['Rec'] = transform(data2['Rec'])
data2['P1'] = transform(data2['P1'])
data2['P2'] = transform(data2['P2'])
data2_labels = data2.pop('Rec')
normed_data2 = norm(data2)
data2_pred = model.predict(normed_data2).flatten()

dataset_path3 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for sequence/DL_verify_EQ_1_1.csv'
data3 = pd.read_csv(dataset_path3)
# data3["IM"] = np.log(data3["IM"])
data3.pop("Time")
# data3.pop("P1")
# data3.pop("P2")
data3['Rec'] = transform(data3['Rec'])
data3['P1'] = transform(data3['P1'])
data3['P2'] = transform(data3['P2'])
data3_labels = data3.pop('Rec')
normed_data3 = norm(data3)
data3_pred = model.predict(normed_data3).flatten()

dataset_path4 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for sequence/DL_verify_EQ_1_3.csv'
data4 = pd.read_csv(dataset_path4)
# data4["IM"] = np.log(data4["IM"])
data4.pop("Time")
# data4.pop("P1")
# data4.pop("P2")
data4['Rec'] = transform(data4['Rec'])
data4['P1'] = transform(data4['P1'])
data4['P2'] = transform(data4['P2'])
data4_labels = data4.pop('Rec')
normed_data4 = norm(data4)
data4_pred = model.predict(normed_data4).flatten()

dataset_path5 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for sequence/DL_verify_EQ_1_5.csv'
data5 = pd.read_csv(dataset_path5)
# data5["IM"] = np.log(data5["IM"])
data5.pop("Time")
# data5.pop("P1")
# data5.pop("P2")
data5['Rec'] = transform(data5['Rec'])
data5['P1'] = transform(data5['P1'])
data5['P2'] = transform(data5['P2'])
data5_labels = data5.pop('Rec')
normed_data5 = norm(data5)
data5_pred = model.predict(normed_data5).flatten()

dataset_path6 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for sequence/DL_verify_EQ_1_7.csv'
data6 = pd.read_csv(dataset_path6)
# data6["IM"] = np.log(data6["IM"])
data6.pop("Time")
# data6.pop("P1")
# data6.pop("P2")
data6['Rec'] = transform(data6['Rec'])
data6['P1'] = transform(data6['P1'])
data6['P2'] = transform(data6['P2'])
data6_labels = data6.pop('Rec')
normed_data6 = norm(data6)
data6_pred = model.predict(normed_data6).flatten()

plt.figure(2)
plt.plot(Time1,invtransform(data1_labels),label='Exact')
plt.plot(Time1,invtransform(data1_pred),label='DNN')
plt.xlabel('Time [days]')
plt.ylabel('Functionality')
plt.xlim([0, 800])
plt.ylim([0, 1])
plt.legend()

plt.figure(3)
plt.plot(Time1,invtransform(data2_labels),label='Exact')
plt.plot(Time1,invtransform(data2_pred),label='DNN')
plt.xlabel('Time [days]')
plt.ylabel('Functionality')
plt.xlim([0, 800])
plt.ylim([0, 1])
plt.legend()

plt.figure(4)
plt.plot(Time1,invtransform(data3_labels),label='Exact')
plt.plot(Time1,invtransform(data3_pred),label='DNN')
plt.xlabel('Time [days]')
plt.ylabel('Functionality')
plt.xlim([0, 800])
plt.ylim([0, 1])
plt.legend()

plt.figure(5)
plt.plot(Time1,invtransform(data4_labels),label='Exact')
plt.plot(Time1,invtransform(data4_pred),label='DNN')
plt.xlabel('Time [days]')
plt.ylabel('Functionality')
plt.xlim([0, 800])
plt.ylim([0, 1])
plt.legend()

plt.figure(6)
plt.plot(Time1,invtransform(data5_labels),label='Exact')
plt.plot(Time1,invtransform(data5_pred),label='DNN')
plt.xlabel('Time [days]')
plt.ylabel('Functionality')
plt.xlim([0, 800])
plt.ylim([0, 1])
plt.legend()

plt.figure(7)
plt.plot(Time1,invtransform(data6_labels),label='Exact')
plt.plot(Time1,invtransform(data6_pred),label='DNN')
plt.xlabel('Time [days]')
plt.ylabel('Functionality')
plt.xlim([4, 800])
plt.ylim([0, 1])
plt.legend()

## Predict the disfunctionality hazard curve

IMe = (np.arange(0.01,2.01,0.01))
IMe.reshape(len(IMe),1)
Tim = (np.arange(4,800,1))
Tim.reshape(len(Tim),1)

Res = np.zeros((len(IMe),len(Tim)))

for ii in np.arange(0,len(IMe),1):
    dat = pd.DataFrame(Tim,columns=['Time'])
    Dis_tim = Kij(IM=IMe[ii],Haz='Eq',time=dat['Time'])
    A1,A2 = Dis_tim.Time()
    dat['P1'] = A1.reshape(len(dat['Time']),1)
    dat['P2'] = A2.reshape(len(dat['Time']),1)
    # dat['IM'] = IMe[ii]*np.ones((len(dat['Time']),1))
    # dat['IM'] = np.log(dat['IM'])
    dat['P1'] = transform(dat['P1'])
    dat['P2'] = transform(dat['P2'])
    dat.pop("Time")
    normed_dat = norm(dat)
    dat_pred = model.predict(normed_dat).flatten()
    Res[ii,:] = 1-invtransform(dat_pred)

    
path11 = '/Users/som/Dropbox/Complex_systems_RNN/Data/EQ_DisfFragility.csv'
DisfFrag_exact = pd.read_csv(path11)
DisfFrag_exact = np.array(DisfFrag_exact)

# for ii in np.arange(0,len(Tim),1):
    


dat_haz = '/Users/som/Dropbox/Complex_systems_RNN/Data/Eq_Haz.csv'
dat_haz = pd.read_csv(dat_haz)

Disf = np.zeros((len(Tim),1))
dat_haz['Haz_diff'] = dat_haz['Haz_diff']/np.abs(np.trapz(dat_haz['IM'],dat_haz['Haz_diff']))

Disf_comp = pd.read_csv('/Users/som/Dropbox/Complex_systems_RNN/Data/Eq_Disf_Exact.csv')

Disf_err = 0
for ii in np.arange(0,len(Tim),1):
    # Disf[ii] = np.sum(dat_haz['Haz_diff']*Res[:,ii])*0.01
    Disf[ii] = np.trapz(dat_haz['Haz_diff']*Res[:,ii],IMe)
    if ii>7:
        Disf_err = Disf_err + np.abs(np.log(Disf[ii])-np.log(Disf_comp['Disf'][ii]))

plt.figure(10)
plt.loglog(Tim,Disf,label='DNN')
plt.loglog(Disf_comp['Time'],Disf_comp['Disf'],label='Exact')
plt.xlim([10,1000])
plt.xlabel('Time to full functionality [days]')
plt.ylabel('Frequency of exceedance')
plt.legend()
