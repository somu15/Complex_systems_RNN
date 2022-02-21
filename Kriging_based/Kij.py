#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:22:20 2020

@author: som
"""

from os import sys
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import random
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh

class Kij:
    
    def __init__(self, IM=None, Haz=None, time=None):
        self.IM = IM
        self.Haz = Haz
        self.time = time
        
    # def Damage(self, IM_sca):
    #     if self.Haz == 'Eq':
            
    #     else:
    #         path = '/Users/som/Dropbox/Complex_systems_RNN/Data/Hurr_DS.csv'
        
    
    def Time(self):
        Times = np.zeros((4,2))
        if self.Haz == 'Eq':
            path = '/Users/som/Dropbox/Complex_systems_RNN/Data/Eq_DS.csv'
            dataset = pd.read_csv(path)
            IM_sca = np.round(self.IM,2)
            Prob_dam = np.zeros((4,1))
            index1 = np.where(dataset['IM']==IM_sca)
            Prob_dam[0] = (1-dataset['Minor'][index1[0]])
            Prob_dam[1] = (dataset['Minor'][index1[0]]-dataset['Mod'][index1[0]])
            Prob_dam[2] = (dataset['Mod'][index1[0]]-dataset['Sev'][index1[0]])
            Prob_dam[3] = (dataset['Sev'][index1[0]])
            Times[0,0] = 1e-3
            Times[0,1] = 1e-3
            Times[1,0] = 2
            Times[1,1] = 3
            Times[2,0] = 30
            Times[2,1] = 90
            Times[3,0] = 90
            Times[3,1] = 270
            
            tot_time1 = 0
            tot_time2 = 0
            for jj in np.arange(0,4,1):
                # tot_time1 = tot_time1 + lognorm.cdf((np.log(self.time)-np.log(Times[jj,0])),0.75)*Prob_dam[jj]
                # tot_time2 = tot_time2 + lognorm.cdf((np.log(self.time)-np.log(Times[jj,1])),0.75)*Prob_dam[jj]
                tot_time1 = tot_time1 + norm.cdf((np.log(self.time)-np.log(Times[jj,0]))/0.75)*Prob_dam[jj]
                tot_time2 = tot_time2 + norm.cdf((np.log(self.time)-np.log(Times[jj,1]))/0.75)*Prob_dam[jj]
            return tot_time1, tot_time2
        elif self.Haz == 'Hurr':
            path = '/Users/som/Dropbox/Complex_systems_RNN/Data/Hurr_DS.csv'
            dataset = pd.read_csv(path)
            IM_sca = np.round(self.IM,2)
            Prob_dam = np.zeros((4,1))
            index1 = np.where(dataset['IM']==IM_sca)
            Prob_dam[0] = (1-dataset['Minor'][index1[0]])
            Prob_dam[1] = (dataset['Minor'][index1[0]]-dataset['Mod'][index1[0]])
            Prob_dam[2] = (dataset['Mod'][index1[0]]-dataset['Sev'][index1[0]])
            Prob_dam[3] = (dataset['Sev'][index1[0]])
            Times[0,0] = 1e-3
            Times[0,1] = 1e-3
            Times[1,0] = 5
            Times[1,1] = 1e-3
            Times[2,0] = 240
            Times[2,1] = 120
            Times[3,0] = 540
            Times[3,1] = 1e-3
            
            tot_time1 = 0
            tot_time2 = 0
            for jj in np.arange(0,4,1):
                tot_time1 = tot_time1 + rayleigh.cdf(self.time,loc=0,scale=(0.8*Times[jj,0]))*Prob_dam[jj]#/(0.8*Times[jj,0])
                tot_time2 = tot_time2 + rayleigh.cdf(self.time,loc=0,scale=(0.8*Times[jj,1]))*Prob_dam[jj]#/(0.8*Times[jj,1])
            return tot_time1, tot_time2
        else:
            print('Not a defined hazard.')
            
            
        
        
        