#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:43:50 2020

@author: imran
"""

import pandas as pd

import os

i= 0
assigned_index = []

list_of_df = []

dataf = ['z', 'y', 'u', 'v', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj']

d = {}
for name in dataf:

    d[name] = pd.DataFrame()
    

   
for dirpaths, dirnames, filenames in os.walk('/home/imran/Desktop/Project HAR/DATASET_pre'):

    i=i+1

    for file in filenames:
        path = os.path.join(dirpaths, file)
        d[dataf[i]] = pd.concat([d[dataf[i]], pd.read_csv(path, sep='\t', encoding='utf-8')], ignore_index=True)
        assigned_index.append(i)
    
    '''assigned_index = set(assigned_index)
    assigned_index = list(assigned_index)
    
    print(dirpaths)
    
    for indexes in assigned_index:'''
    
    print(dirpaths)

    path_i = os.path.join(dirpaths,  dataf[i]+'.csv')
    
    d[dataf[i]].to_csv(path_i,encoding='utf-8', index=False)
    

        