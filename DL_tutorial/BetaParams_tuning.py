
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

dataset_path = '/Users/som/Dropbox/Complex_systems_RNN/Data/DL_test_data_Eq_Hazus_125.csv'

count1 = 0
co1 = 0
ERRVALS = np.zeros((50,50))
param_a = np.zeros((50,1))
param_b = np.zeros((50,1))
for ii1 in np.arange(0.1,100,2):
    co2 = 0
    for jj1 in np.arange(0.1,100,2):
        np.random.seed(100)
        random.set_seed(100)
        
        dataset = pd.read_csv(dataset_path)
        dataset.pop("IM")
        dataset.pop("Time")

        a1 = ii1
        b1 = jj1
        loc1 = 0
        sca1 = 1
        
        def transform(Rec):
            # r1 = Rec == 0.0
            # r2 = Rec == 1.0
            # Rec[r1] = 0.000000001
            # Rec[r2] = 0.999999999
            # Rec = -np.log(1-Rec)*2.5
            Rec = beta.ppf(Rec,a1,b1,loc1,sca1)
            return Rec
        
        def invtransform(x):
            return beta.cdf(x,a1,b1,loc1,sca1)
            # return (1-np.exp(-x/2.5))
        
        dataset['Rec'] = transform(dataset['Rec'])
        
        dataset['P1'] = transform(dataset['P1'])
        
        dataset['P2'] = transform(dataset['P2'])
        
        dataset.tail()
        
        train_dataset = dataset.sample(frac=0.8,random_state=100)
        test_dataset = dataset.drop(train_dataset.index)
        
        # Inspect the data
        
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
            layers.Dense(2, activation='softmax',bias_initializer='zeros'),
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
        
        
        EPOCHS = 1500
        
        history = model.fit(
          normed_train_data, train_labels,
          epochs=EPOCHS, validation_split = 0.0, verbose=0,
          callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)
        
        
        
        # Make predictions
        
        test_predictions = model.predict(normed_test_data).flatten()
        
        ## Verify model for multiple recovery curves
        
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
            # plt.plot(dat['Time'],sigm(dat_pred))
            # plt.xlim([0, 800])
            # plt.ylim([0, 1])
            
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
            
        ERRVALS[co1,co2] = Disf_err
        param_b[co2] = jj1
        co2 = co2 + 1
    param_a[co1] = ii1
    co1 = co1 + 1


ind = np.unravel_index(np.argmin(ERRVALS, axis=None), ERRVALS.shape)

# dataset_path = '/Users/som/Dropbox/Complex_systems_RNN/Data/DL_test_data_Eq_Hazus_125.csv'

# count1 = 0
# co1 = 0
# ERRVALS = np.zeros((200,1))
# param_sto = np.zeros((200,1))
# for ii1 in np.arange(0.01,100.0,0.5):
#     np.random.seed(100)
#     random.set_seed(100)
        
#     dataset = pd.read_csv(dataset_path)
#     dataset.pop("IM")
#     dataset.pop("Time")

#     a1 = ii1
#     loc1 = 0
#     sca1 = 1
    
#     def transform(Rec):
#         # r1 = Rec == 0.0
#         # r2 = Rec == 1.0
#         # Rec[r1] = 0.000000001
#         # Rec[r2] = 0.999999999
#         # Rec = -np.log(1-Rec)*2.5
#         Rec = beta.ppf(Rec,a1,a1,loc1,sca1)
#         return Rec
    
#     def invtransform(x):
#         return beta.cdf(x,a1,a1,loc1,sca1)
#         # return (1-np.exp(-x/2.5))
    
#     dataset['Rec'] = transform(dataset['Rec'])
    
#     dataset['P1'] = transform(dataset['P1'])
    
#     dataset['P2'] = transform(dataset['P2'])
    
#     dataset.tail()
    
#     train_dataset = dataset.sample(frac=0.8,random_state=100)
#     test_dataset = dataset.drop(train_dataset.index)
    
#     # Inspect the data
    
#     train_stats = train_dataset.describe()
#     train_stats.pop("Rec")
#     train_stats = train_stats.transpose()
#     train_stats
    
#     # Split features from labels
    
#     train_labels = train_dataset.pop('Rec')
#     test_labels = test_dataset.pop('Rec')
    
#     # Normalize the data
    
#     def norm(x):
#       return (x - train_stats['mean']) / train_stats['std']
#     normed_train_data = norm(train_dataset)
#     normed_test_data = norm(test_dataset)
    
#     # Build the model ,kernel_regularizer='l2'
    
#     def build_model():
#       model = keras.Sequential([
#         layers.Dense(4, activation='softmax', input_shape=[len(train_dataset.keys())],bias_initializer='zeros'),
#         layers.Dense(4, activation='softmax',bias_initializer='zeros'),
#         layers.Dense(1,bias_initializer='zeros')
#       ])
    
#       # optimizer = tf.keras.optimizers.RMSprop(0.001)
#       optimizer = tf.keras.optimizers.RMSprop(0.001)
    
#       model.compile(loss='mse',
#                     optimizer=optimizer,
#                     metrics=['mae', 'mse'])
#       return model
    
#     model = build_model()
    
#     # Inspect the model
    
#     model.summary()
    
    
#     EPOCHS = 1500
    
#     history = model.fit(
#       normed_train_data, train_labels,
#       epochs=EPOCHS, validation_split = 0.0, verbose=0,
#       callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)
    
    
    
#     # Make predictions
    
#     test_predictions = model.predict(normed_test_data).flatten()
    
#     ## Verify model for multiple recovery curves
    
#     ## Predict the disfunctionality hazard curve
    
#     IMe = (np.arange(0.01,2.01,0.01))
#     IMe.reshape(len(IMe),1)
#     Tim = (np.arange(4,800,1))
#     Tim.reshape(len(Tim),1)
    
#     Res = np.zeros((len(IMe),len(Tim)))
    
#     for ii in np.arange(0,len(IMe),1):
#         dat = pd.DataFrame(Tim,columns=['Time'])
#         Dis_tim = Kij(IM=IMe[ii],Haz='Eq',time=dat['Time'])
#         A1,A2 = Dis_tim.Time()
#         dat['P1'] = A1.reshape(len(dat['Time']),1)
#         dat['P2'] = A2.reshape(len(dat['Time']),1)
#         # dat['IM'] = IMe[ii]*np.ones((len(dat['Time']),1))
#         # dat['IM'] = np.log(dat['IM'])
#         dat['P1'] = transform(dat['P1'])
#         dat['P2'] = transform(dat['P2'])
#         dat.pop("Time")
#         normed_dat = norm(dat)
#         dat_pred = model.predict(normed_dat).flatten()
#         Res[ii,:] = 1-invtransform(dat_pred)
#         # plt.plot(dat['Time'],sigm(dat_pred))
#         # plt.xlim([0, 800])
#         # plt.ylim([0, 1])
        
#     dat_haz = '/Users/som/Dropbox/Complex_systems_RNN/Data/Eq_Haz.csv'
#     dat_haz = pd.read_csv(dat_haz)
    
#     Disf = np.zeros((len(Tim),1))
#     dat_haz['Haz_diff'] = dat_haz['Haz_diff']/np.abs(np.trapz(dat_haz['IM'],dat_haz['Haz_diff']))
    
#     Disf_comp = pd.read_csv('/Users/som/Dropbox/Complex_systems_RNN/Data/Eq_Disf_Exact.csv')
    
#     Disf_err = 0
#     for ii in np.arange(0,len(Tim),1):
#         # Disf[ii] = np.sum(dat_haz['Haz_diff']*Res[:,ii])*0.01
#         Disf[ii] = np.trapz(dat_haz['Haz_diff']*Res[:,ii],IMe)
#         if ii>7:
#             Disf_err = Disf_err + np.abs(np.log(Disf[ii])-np.log(Disf_comp['Disf'][ii]))
        
#     ERRVALS[co1] = Disf_err
#     param_sto[co1] = a1
#     co1 = co1 + 1

