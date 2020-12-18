#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:23:00 2020

@author: imran
"""

import pandas as pd

df1 = pd.read_csv('Standing_final/Standing_hand_accelerometer.csv')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Standing_final/Standing_hand_gyroscope.csv' )
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Standing_final/Standing_pocket_accelerometer.csv')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Standing_final/Standing_pocket_gyroscope.csv')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']

Standing_df = pd.concat([df1,df2,df3,df4], axis=1)
Standing_df = Standing_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Standing')
Standing_df['Activity'] = list_1


Standing_df.to_csv('Standing_dataset_full.csv')







df1 = pd.read_csv('Walking_final/Walking_hand_accelerometer.csv')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Walking_final/Walking_hand_gyroscope.csv' )
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Walking_final/Walking_pocket_accelerometer.csv')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Walking_final/Walking_pocket_gyroscope.csv')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Walking_df = pd.concat([df1,df2,df3,df4], axis=1)
Walking_df = Walking_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Walking')
Walking_df['Activity'] = list_1

Walking_df.to_csv('Walking_dataset_full.csv')



df1 = pd.read_csv('Sitting_final/Sitting_Hand_accelerometer.csv')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Sitting_final/Sitting_Hand_gyroscope.csv')
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Sitting_final/Sitting_Pocket_accelerometer.csv')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Sitting_final/Sitting_Pocket_gyroscope.csv')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Sitting_df = pd.concat([df1,df2,df3,df4], axis=1)
Sitting_df = Sitting_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Sitting')
Sitting_df['Activity'] = list_1

Sitting_df.to_csv('Sitting_dataset_full.csv')




df1 = pd.read_csv('Downstair_final/Down_hand_accelerometer.csv')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Downstair_final/Down_hand_gyroscope.csv')
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Downstair_final/Down_pocket_accelerometer.csv')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Downstair_final/Down_pocket_gyroscope.csv')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Downstair_df = pd.concat([df1,df2,df3,df4], axis=1)
Downstair_df = Downstair_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Downstair')
Downstair_df['Activity'] = list_1

Downstair_df.to_csv('Downstair_dataset_full.csv')









df1 = pd.read_csv('Upstair_final/Up_hand_accelerometer.csv')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Upstair_final/Up_hand_gyroscope.csv')
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Upstair_final/Up_pocket_accelerometer.csv')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Upstair_final/Up_pocket_gyroscope.csv')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Upstair_df = pd.concat([df1,df2,df3,df4], axis=1)
Upstair_df = Upstair_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Upstair')
Upstair_df['Activity'] = list_1

Upstair_df.to_csv('Upstair_dataset_full.csv')




df1 = pd.read_csv('Car_final/Car_Hand_accelerometer.csv')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Car_final/Car_Hand_gyroscope.csv')
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Car_final/Car_Pocket _accelerometer.csv')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Car_final/Car_Pocket _gyroscope.csv')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Car_df = pd.concat([df1,df2,df3,df4], axis=1)
Car_df = Car_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Sitting_Car')
Car_df['Activity'] = list_1

Car_df.to_csv('Car_dataset_full.csv')




df1 = pd.read_csv('Cycling_final/cycling_hand_accelerometer.csv', sep='\t')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Cycling_final/cycling_hand_gyroscope.csv', sep='\t')
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Cycling_final/cycling_pocket_accelerometer.csv', sep='\t')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Cycling_final/cycling_pocket_gyroscope.csv', sep='\t')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Cycling_df = pd.concat([df1,df2,df3,df4], axis=1)
Cycling_df = Cycling_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Cycling')
Cycling_df['Activity'] = list_1

Cycling_df.to_csv('Cycling_dataset_full.csv')



df1 = pd.read_csv('Jogging_final/Jogging_hand_accelerometer.csv', sep='\t')
df1 = df1.iloc[:, 1:4]
df1.columns = ['h_ac_X', 'h_ac_Y', 'h_ac_Z']
df2 = pd.read_csv('Jogging_final/Jogging_hand_gyroscope.csv', sep='\t')
df2 = df2.iloc[:, 1:4]
df2.columns = ['h_gy_X', 'h_gy_Y', 'h_gy_Z']
df3 = pd.read_csv('Jogging_final/Jogging_pocket_accelerometer.csv', sep='\t')
df3 = df3.iloc[:, 1:4]
df3.columns = ['p_ac_X', 'p_ac_Y', 'p_ac_Z']
df4 = pd.read_csv('Jogging_final/Jogging_pocket_gyroscope.csv', sep='\t')
df4 = df4.iloc[:, 1:4]
df4.columns = ['p_gy_X', 'p_gy_Y', 'p_gy_Z']


Jogging_df = pd.concat([df1,df2,df3,df4], axis=1)
Jogging_df = Jogging_df.iloc[0:44000, :]

list_1 = []
for i in range(0, 44000):
    list_1.append('Jogging')
Jogging_df['Activity'] = list_1

Jogging_df.to_csv('Jogging_dataset_full.csv')




Final_dataset = pd.DataFrame()

Final_dataset = pd.concat([Walking_df, Jogging_df, Standing_df, Upstair_df, Downstair_df, Sitting_df, Car_df, Cycling_df], axis=0, ignore_index=True)

Final_dataset.to_csv('Final_Dataset_V3.csv', index=None)