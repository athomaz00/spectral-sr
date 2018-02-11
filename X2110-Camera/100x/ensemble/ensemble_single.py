# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:20:55 2018

@author: athomaz
"""

from lmfit import Parameters
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel, LognormalModel, ExponentialGaussianModel,SkewedGaussianModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def  lorentzian(x, A, mu, sigma):
    return (A/np.pi)*(sigma/((x-mu)**2 + sigma**2))

def pvoight(x, A, mu, sigma, fraction):
    return (1.0-fraction)*(A/((sigma/np.sqrt(2*np.log(2)))*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))+ fraction*(A/np.pi)*(sigma/((x-mu)**2 + sigma**2))
def gaussian(x, A, mu, sigma):
    return (A/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))

file = ['bQD605-532nm-exc.xls', 'sQD620-532nm-exc.xls', 'bQD705-532nm-exc.xls' ]

ensemble = pd.read_excel(file[1])

x = ensemble.iloc[130:340,0].values
y = ensemble.iloc[130:340,1].values/np.max(ensemble.iloc[130:340,1].values)

#x = ensemble.iloc[120:680,0].values
#y = ensemble.iloc[120:680,1].values/np.max(ensemble.iloc[120:680,1].values)

plt.figure()
plt.plot(x, y)




gmodel = SkewedGaussianModel()
parsG = gmodel.guess(y, x=x)
fitG  = gmodel.fit(y, parsG, x=x)
plt.plot(x, fitG.best_fit, 'r-')
print(fitG.fit_report(min_correl=0.25))


yl = lorentzian(x, 58.773057295944994, 611.715642,  19.3790430)
plt.plot(x, yl, 'g-')
plt.xlim(550,660)