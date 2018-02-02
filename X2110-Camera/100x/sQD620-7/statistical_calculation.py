# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:09:14 2018

@author: athomaz
"""

import pandas as pd


file = 'center-sigma-1.xlsx'

specs = pd.read_excel(file)

for i, row in specs.iterrows():
    print(row)