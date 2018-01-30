# -*- coding: utf-8 -*-
# =============================================================================
# This code does the function fitting (Gaussian or Lorentzian) using the Lmfit 
# package. 
# Input: outuput xlsx file from wavelength_calculation.py at variable file
# Output: xlsx file with fittings parameters: 
# - Type of fitting
# - Parameters
# - Chi square
# - If output is 0,0,0 it means chi square for the fitting is less than 0.069
# Created on Mon Jan 29 18:17:55 2018
# 
# @author: Andre Thomaz
# =============================================================================

from lmfit.models import GaussianModel, LorentzianModel, SkewedGaussianModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# =============================================================================
def fitting(x,y):
     gmodel = GaussianModel()
     parsG = gmodel.guess(y, x=x)
     fitG  = gmodel.fit(y, parsG, x=x)
     chiG = fitG.chisqr
     #print(out.fit_report(min_correl=0.25))
     plt.figure()
     plt.plot(x,y, 'o')
     plt.plot(x, fitG.best_fit, 'r-')
     print(chiG,i)
     
     lmodel = LorentzianModel()
     parsL = lmodel.guess(y, x=x)
     fitL = lmodel.fit(y, parsL, x=x )
     chiL = fitL.chisqr
     plt.figure()
     plt.plot(x,y, 'o')
     plt.plot(x, fitL.best_fit, 'r-')
     print(chiL,i)
     
     if chiG < chiL and 0.0<chiG<0.055 :
         return ['GaussianModel',fitG.values, chiG]
     elif 0.0<chiL<0.055:
         return ['lorentzian',fitL.values, chiL]
     else:
         return [0, 0, 0]
 
# =============================================================================

file = 'output-6.xlsx'

specs = pd.read_excel(file)

columns = len(specs.columns)

fittingTable =[]

i=0
while i<int(columns): 
    x = specs.iloc[7:33,i].values
    y = specs.iloc[7:33,i+2].values
    
    fit = fitting(x,y)
    fittingTable.append(fit)
    i += 3

df = pd.DataFrame(fittingTable, columns=['fitting-function', 'values', 'chisq'])

sigmaTable = []
centerTable = []

for i, row in df.iterrows():
    if row['values'] != 0:
        sigmaTable.append(row['values']['sigma'])
        centerTable.append(row['values']['center'])
        
centers_sigma = np.array([centerTable, sigmaTable])
centers_sigma = centers_sigma.T

centers_sigma = pd.DataFrame(centers_sigma, columns=['centers', 'sigmas'])

fileName = file.split('-')

#writer = pd.ExcelWriter('fitting-' + fileName[1])
#df.to_excel(writer,'Sheet1')
writer = pd.ExcelWriter('center-sigma-green-' + fileName[1])
centers_sigma.to_excel(writer,'Sheet1')
writer.save()






