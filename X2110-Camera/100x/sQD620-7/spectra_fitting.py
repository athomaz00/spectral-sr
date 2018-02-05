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
     plt.figure()
     plt.plot(x,y, 'o')
     plt.plot(x, fitG.best_fit, 'g-')
     print(chiG,i)
     
     lmodel = LorentzianModel()
     try:    
         parsL = lmodel.guess(y, x=x)
     except KeyError:
         parsL = lmodel.make_params(sigma=30, center=605.0, amplitude=1.0) 
     fitL = lmodel.fit(y, parsL, x=x )
     chiL = fitL.chisqr
     plt.figure()
     plt.plot(x,y, 'o')
     plt.plot(x, fitL.best_fit, 'b-')
     print(chiL,i)
     
     vmodel = PseudoVoigtModel()
     paramsV = Parameters()
     paramsV.add('sigma', value=30)
     paramsV.add('center', value=605.0)
     paramsV.add('amplitude', value=1.0)
     paramsV.add('fraction', value=0.2, max=1.0)
     paramsV.add('fwhm', value=20)
     try: 
         parsV = vmodel.guess(y, x=x)
     except KeyError:
         parsV = vmodel.make_params(sigma=30, center=605.0, amplitude=1.0, fraction=0.5) 
     fitV = vmodel.fit(y, parsV, x=x )
     popt, pcov = curve_fit(pvoight, x, y, p0=[70.0, 605.0, 30.0, 0.5], maxfev=20000)
     chiV = fitV.chisqr
     #print(fitV.fit_report(min_correl=0.25))
     plt.figure()
     plt.plot(x,y, 'o')
     plt.plot(x, fitV.best_fit, 'k-')
     plt.plot(x, pvoight(x, *popt), 'r-')
     print(chiV,i)
     print(popt)
     
     if chiG < chiL and 0<chiG<0.05:
         return ['gaussian',fitG.values, chiG]
     elif chiL<chiV and 0<chiL <0.05:
         return ['lorentzian',fitL.values, chiL]
     elif chiV<0.05:
          return ['pvoight',fitV.values, chiV]
     else:
         return [0, 0, 0]
 
# =============================================================================

file = 'output-sQD620-2.xlsx'

specs = pd.read_excel(file)

columns = len(specs.columns)

fittingTable =[]

i=0
while i<int(columns): 
    x = specs.iloc[5:22,i]
    y = specs.iloc[5:22,i+1]
    
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
writer = pd.ExcelWriter('center-sigma-v' + fileName[1])
centers_sigma.to_excel(writer,'Sheet1')
writer.save()






