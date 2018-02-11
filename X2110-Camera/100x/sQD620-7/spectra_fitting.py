# -*- coding: utf-8 -*-
# =============================================================================
# This code does the function fitting (Gaussian or Lorentzian) using the Lmfit 
# package. 
# Input: outuput xlsx file from wavelength_calculation.py at variable file
# Output: xlsx file with fittings parameters: 
# - Type of fitting
# - Parameters
# - Chi square
# - If output is 0,0,0 it means chi square for the fitting is less than 0.05
# Output2: xlsx file with centers and sigmas for each fitting
# Created on Mon Jan 29 18:17:55 2018
# 
# @author: Andre Thomaz
# =============================================================================
from lmfit import Parameters
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel,PseudoVoigtModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def pvoight(x, A, mu, sigma, fraction):
    return (1.0-fraction)*(A/((sigma/np.sqrt(2*np.log(2)))*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))+ fraction*(A/np.pi)*(sigma/((x-mu)**2 + sigma**2))

# =============================================================================
def fitting(x,y):
     gmodel = GaussianModel()
     try:
         parsG = gmodel.guess(y, x=x)
     except KeyError:
         parsG = gmodel.make_params(sigma=30, center=605.0, amplitude=1.0) 
     fitG  = gmodel.fit(y, parsG, x=x)
     chiG = fitG.chisqr
     #print(fitG.fit_report(min_correl=0.25))
#     plt.figure()
#     plt.plot(x,y, 'o')
#     plt.plot(x, fitG.best_fit, 'g-')
#     print(chiG,i)
     
     lmodel = LorentzianModel()
     try:    
         parsL = lmodel.guess(y, x=x)
     except KeyError:
         parsL = lmodel.make_params(sigma=30, center=605.0, amplitude=1.0) 
     fitL = lmodel.fit(y, parsL, x=x )
     chiL = fitL.chisqr
#     plt.figure()
#     plt.plot(x,y, 'o')
#     plt.plot(x, fitL.best_fit, 'b-')
     print(chiL,i)
     

     
     if chiG < chiL and 0<chiG<0.05:
         return ['gaussian',fitG.values, chiG]
     elif  0<chiL <0.05:
         return ['lorentzian',fitL.values, chiL]
     else:
         return [0, 0, 0]
 
# =============================================================================

file = 'series-sQD620-1.xlsx'

specs = pd.read_excel(file)

columns = len(specs.columns)

fittingTable =[]

i=0
while i<int(columns): 
    x = specs.iloc[5:22,i]
    y = specs.iloc[5:22,i+1]/np.max(specs.iloc[5:22,i+1])
    
    fit = fitting(x,y)
    fittingTable.append(fit)
    i += 2

df = pd.DataFrame(fittingTable, columns=['fitting-function', 'values', 'chisq'])

sigmaTable = []
centerTable = []
amplitudeTable = []
fractionTable = []
functionTable = []

for i, row in df.iterrows():
    if row['values'] != 0:
        sigmaTable.append(float(row['values']['sigma']))
        centerTable.append(float(row['values']['center']))
        amplitudeTable.append(float(row['values']['amplitude']))
        if row['fitting-function'] == 'pvoight':
            fractionTable.append(float(row['values']['fraction']))
        else: 
            fractionTable.append(0.0)
        functionTable.append(row['fitting-function'])
        
centers_sigma = np.array([centerTable, sigmaTable, amplitudeTable, fractionTable, functionTable])
centers_sigma = centers_sigma.T

centers_sigma = pd.DataFrame(centers_sigma, columns=['centers', 'sigmas', 'amplitude', 'fraction', 'function'])



fileName = file.split('output')

#writer = pd.ExcelWriter('fitting-' + fileName[1])
#df.to_excel(writer,'Sheet1')
writer = pd.ExcelWriter('center-sigma-s' + fileName[1])
centers_sigma.to_excel(writer,'Sheet1')
writer.save()






