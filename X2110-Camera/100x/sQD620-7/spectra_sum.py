# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:14:38 2018

@author: athomaz
"""

from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ['center-sigma-v-sQD620-1.xlsx', 'center-sigma-v-sQD620-2.xlsx', 'center-sigma-v-sQD620-3.xlsx', 'center-sigma-v-sQD620-5.xlsx']

def  lorentzian(x, A, mu, sigma):
    return (A/np.pi)*(sigma/((x-mu)**2 + sigma**2))

def gaussian(x, A, mu, sigma):
    return (A/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))

def pvoight(x, A, mu, sigma, fraction):
    return (1.0-fraction)*(A/((sigma/np.sqrt(2*np.log(2)))*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))+ fraction*(A/np.pi)*(sigma/((x-mu)**2 + sigma**2))



x = np.linspace(560,650)

df = pd.DataFrame()

for file in files:
    d_temp = pd.read_excel(file)
    df = df.append(d_temp)

specs = []    

for i,row in df.iterrows():
    if row['function'] == 'lorentzian':
        yl = lorentzian(x, row['amplitude'], row['centers'], row['sigmas'])
        specs.append(yl)
        plt.plot(x, yl)
    elif row['function'] == 'gaussian':
        yg = gaussian(x, row['amplitude'], row['centers'], row['sigmas'])
        specs.append(yg)
        plt.plot(x, gaussian(x, row['amplitude'], row['centers'], row['sigmas']))
    elif row['function'] == 'pvoight':
        yv = pvoight(x, row['amplitude'], row['centers'], row['sigmas'], row['fraction'])
        specs.append(yv)
        plt.plot(x, pvoight(x, row['amplitude'], row['centers'], row['sigmas'], row['fraction']))
        
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity (a.u.)')


center_avg = df['centers'].mean()
center_std = df['centers'].std()
center_err = center_std/np.sqrt(df.shape[0])
center_disp = center_std*100/center_avg

width_avg = df['sigmas'].mean()
width_std = df['sigmas'].std()
width_err = width_std /np.sqrt(df.shape[0])
width_disp = width_std*100/width_avg

print('Number of sQD620 = ' + str(df.shape[0]))
print('=========================================')
print('Center AVG = '+str(center_avg) + '+-'+str(center_err))
print('Center STD = '+str(center_std))
print('Center Dispersion = ' + str(center_disp))
print('=========================================')
print('Width AVG = '+str(width_avg) + '+-'+str(width_err))
print('Width STD = '+str(width_std))
print('Width Dispersion = ' + str(width_disp))
print('=========================================')




specs=np.array(specs)
specs = specs.T
specsSum = np.sum(specs, axis=1)
ys = specsSum# /np.max(specsSum)
ys = ys/np.max(ys)
plt.plot(x, ys, '--r')

lmodel = LorentzianModel()
parsL = lmodel.guess(ys, x=x)
fitL = lmodel.fit(ys, parsL, x=x)
chiL = fitL.chisqr

gmodel = GaussianModel()
parsG = gmodel.guess(ys, x=x)
fitG = gmodel.fit(ys, parsG, x=x)
chiG = fitG.chisqr

vmodel = PseudoVoigtModel()
parsV = vmodel.guess(ys, x=x)
fitV = vmodel.fit(ys, parsV, x=x)
chiV = fitV.chisqr



plt.figure()
plt.plot(x, ys, '--r')
plt.plot(x, fitL.best_fit, 'g')
plt.plot(x, fitG.best_fit, 'k')
plt.plot(x, fitV.best_fit, 'b')

if chiL<chiG and chiL<chiV:
    print(fitL.best_values)
    print(fitL.fit_report(min_correl=0.25))

elif chiG<chiV:
    print(fitG.best_values)
    print(fitG.fit_report(min_correl=0.25))
else:
    print(fitV.best_values)
    print(fitV.fit_report(min_correl=0.25))

    
    
