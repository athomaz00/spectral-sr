# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:30:09 2018

@author: athomaz
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

files = ['output-bQD705-1.xlsx','output-bQD705-2.xlsx','output-bQD705-3.xlsx','output-bQD705-4.xlsx','output-bQD705-5.xlsx',]


spectralMean =[]



for file in files:
    d_temp = pd.read_excel(file)
  
    columns = len(d_temp.columns)
    i=0
    while i<int(columns): 
        x = d_temp.iloc[5:22,i]
        y = d_temp.iloc[5:22,i+1]
        spectralMeanRed = np.sum(np.multiply(x,y))/np.sum(y)
        spectralMean.append(spectralMeanRed)
        i += 2
        
plt.figure(3)

plt.errorbar([705.0], np.mean(spectralMean), yerr=np.std(spectralMean), marker='o', markersize=5)
plt.xlim(570,750)
plt.ylim(560,720)
plt.grid('on')
plt.xlabel('Nominal Wavelength (nm)')
plt.ylabel('Spectral Mean (nm)')

   