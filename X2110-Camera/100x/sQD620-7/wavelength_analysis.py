# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:40:19 2018

@author: athomaz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ['series-sQD620-1.xlsx'] #, 'output-2.xlsx', 'output-3.xlsx', 'output-4.xlsx', 'output-5.xlsx']

#df = pd.read_excel(files[0])

def plot_graph(file):
    df = pd.read_excel(file)
    i=0
   
    while i < len(df.columns):
        
        red_col = i+1

        plt.figure(1, figsize=(4,8))
        if i not in np.arange(0,520,5):
            plt.plot(df.iloc[:,i],df.iloc[:,red_col]/np.max(df.iloc[:,red_col]))
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized Intensity (a.u.)')
            plt.title('Green Dye and Red Dye from Tetraspeck Beads')
            #plt.xlim(640,750)
        i += 2
        
def counter(files):
    counter = 0
    for file in files:
        df = pd.read_excel(file)
        counter += int(len(df.columns)/3)
    print("Number of spectra " + str(counter))
    return counter
    

        
for file in files:
    plot_graph(file)

counter(files)
    