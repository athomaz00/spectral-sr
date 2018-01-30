# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:17:55 2018

@author: athomaz
"""

from lmfit.models import GaussianModel, LorentzianModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



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
    
    if chiG < chiL and chiG<0.05:
        return ['gaussian',fitG.values, chiG]
    elif chiL <0.05:
        return ['lorentzian',fitL.values, chiL]


file = pd.read_excel('output-2.xlsx')

columns = len(file.columns)

fittingTable =[]

i=0
while i<int(columns): 
    x = file.iloc[4:24,i]
    y = file.iloc[4:24,i+1]
    
    fit = fitting(x,y)
    fittingTable.append(fit)
    i += 2