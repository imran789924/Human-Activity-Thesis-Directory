#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:23:00 2020

@author: imran
"""

import pandas as pd

df1 = pd.read_csv('Standing_final/Standing_hand_accelerometer.csv')
df1 = df1.iloc[:, 2:5]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Standing_final/Standing_hand_gyroscope.csv' )
df2 = df2.iloc[:, 2:5]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Standing_final/Standing_pocket_accelerometer.csv')
df3 = df3.iloc[:, 2:5]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Standing_final/Standing_pocket_gyroscope.csv')
df4 = df4.iloc[:, 2:5]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Standing_df = pd.concat([df1,df2,df3,df4], axis=1)

Standing_df = Standing_df.iloc[0:32050, :]

Standing_df.to_csv('Standing_dataset_full.csv')




df1 = pd.read_csv('Walking_final/Walking_hand_accelerometer.csv')
df1 = df1.iloc[:, 2:5]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Walking_final/Walking_hand_gyroscope.csv' )
df2 = df2.iloc[:, 2:5]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Walking_final/Walking_pocket_accelerometer.csv')
df3 = df3.iloc[:, 2:5]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Walking_final/Walking_pocket_gyroscope.csv')
df4 = df4.iloc[:, 2:5]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Walking_df = pd.concat([df1,df2,df3,df4], axis=1)

Walking_df = Walking_df.iloc[0:32050, :]

Walking_df.to_csv('Walking_dataset_full.csv')




df1 = pd.read_csv('Sitting_final/Sitting_Hand_accelerometer.csv', sep='\t')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Sitting_final/Sitting_Hand_gyroscope.csv', sep='\t')
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Sitting_final/Sitting_Pocket_accelerometer.csv', sep='\t')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Sitting_final/Sitting_Pocket_gyroscope.csv', sep='\t')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Sitting_df = pd.concat([df1,df2,df3,df4], axis=1)

Sitting_df = Sitting_df.iloc[0:32050, :]

Sitting_df.to_csv('Standing_dataset_full.csv')


list_1 = []

for i in range(0, 32050):
    list_1.append('Sitting')


Sitting_df['Activity'] = list_1


Sitting_df.isnull().sum()

final_df = pd.read_csv('Final_Dataset.csv')

Final_dataset = pd.concat([Sitting_df, final_df], axis=0, ignore_index=True)


Final_dataset.to_csv('Final_Dataset_V1.csv')