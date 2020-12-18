#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 20:22:25 2020

@author: imran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


df = pd.read_csv('Final_Dataset_V3.csv')
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]



scaler = RobustScaler()
X_S = scaler.fit_transform(X)


#Individual plotting

fig = plt.figure(figsize=(4,10))
ax = fig.add_subplot(211, projection='3d')
ax.scatter(xs=X_S[310000:312000, 0], ys=X_S[310000:312000, 1], zs=X_S[310000:312000, 2])
ax.set_xlim(-10,+10)
ax.set_ylim(-10,+10)
ax.set_zlim(-10,+10)
ax.set_title('Hand')

ax = fig.add_subplot(212, projection='3d')
ax.scatter(xs=X_S[310000:312000, 6], ys=X_S[310000:312000, 7], zs=X_S[310000:312000, 8])
ax.set_xlim(-10,+10)
ax.set_ylim(-10,+10)
ax.set_zlim(-10,+10)
ax.set_title('Pocket')
plt.show()





#Comparative analysis, 2 accelerometer, 2 gyroscope

#Accelerometer Hand
fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X_S[2000:2500, 0], ys=X_S[2000:2500, 1], zs=X_S[2000:2500, 2], c='red', alpha=0.4)
ax.scatter(xs=X_S[46000:46100, 0], ys=X_S[46000:46100, 1], zs=X_S[46000:46100, 2], c='green', alpha=0.6)
ax.scatter(xs=X_S[90000:92500, 0], ys=X_S[90000:92500, 1], zs=X_S[90000:92500, 2], c='blue')
ax.scatter(xs=X_S[140000:140500, 0], ys=X_S[140000:140500, 1], zs=X_S[140000:140500, 2], c='y')
ax.scatter(xs=X_S[180000:180500, 0], ys=X_S[180000:180500, 1], zs=X_S[180000:180500, 2], c='c')
ax.scatter(xs=X_S[222000:223500, 0], ys=X_S[222000:223500, 1], zs=X_S[222000:223500, 2], c='black')
ax.scatter(xs=X_S[270000:270500, 0], ys=X_S[270000:270500, 1], zs=X_S[270000:270500, 2], c='magenta')
ax.scatter(xs=X_S[310000:310300, 0], ys=X_S[310000:310300, 1], zs=X_S[310000:310300, 2], c='brown')
#ax.set_xlim(-10,+10)
#ax.set_ylim(-10,+10)
#ax.set_zlim(-10,+10)
ax.set_title("Accelerometer Hand(Wrist)")
ax.legend(['Walking', 'Jogging', 'Standing', 'Upstair', 'Downstair', 'Sitting', 'Sitting_Car', 'Cycling'])
plt.show()



#Accelerometer Pocket
fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X_S[2000:2500, 6], ys=X_S[2000:2500, 7], zs=X_S[2000:2500, 8], c='red', alpha=0.4)
ax.scatter(xs=X_S[46000:46100, 6], ys=X_S[46000:46100, 7], zs=X_S[46000:46100, 8], c='green', alpha=0.7)
ax.scatter(xs=X_S[90000:90500, 6], ys=X_S[90000:90500, 7], zs=X_S[90000:90500, 8], c='blue')
ax.scatter(xs=X_S[140000:140500, 6], ys=X_S[140000:140500, 7], zs=X_S[140000:140500, 8], c='y')
ax.scatter(xs=X_S[180000:180300, 6], ys=X_S[180000:180300, 7], zs=X_S[180000:180300, 8], c='c')
ax.scatter(xs=X_S[222000:222900, 6], ys=X_S[222000:222900, 7], zs=X_S[222000:222900, 8], c='black')
ax.scatter(xs=X_S[270000:270500, 6], ys=X_S[270000:270500, 7], zs=X_S[270000:270500, 8], c='magenta')
ax.scatter(xs=X_S[310000:310500, 6], ys=X_S[310000:310500, 7], zs=X_S[310000:310500, 8], c='brown')
#ax.set_xlim(-10,+10)
#ax.set_ylim(-10,+10)
#ax.set_zlim(-10,+10)
ax.set_title('Accelerometer Pocket(Waist)')
ax.legend(['Walking', 'Jogging', 'Standing', 'Upstair', 'Downstair', 'Sitting', 'Sitting_Car', 'Cycling'])
plt.show()



#Gyroscope Hand
fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X_S[2000:2500, 3], ys=X_S[2000:2500, 4], zs=X_S[2000:2500, 5], c='red', alpha=0.4)
ax.scatter(xs=X_S[46000:46100, 3], ys=X_S[46000:46100, 4], zs=X_S[46000:46100, 5], c='green', alpha=0.7)
ax.scatter(xs=X_S[90000:90500, 3], ys=X_S[90000:90500, 4], zs=X_S[90000:90500, 5], c='blue')
ax.scatter(xs=X_S[140000:140500, 3], ys=X_S[140000:140500, 4], zs=X_S[140000:140500, 5], c='y')
ax.scatter(xs=X_S[180000:180500, 3], ys=X_S[180000:180500, 4], zs=X_S[180000:180500, 5], c='c')
ax.scatter(xs=X_S[222000:223000, 3], ys=X_S[222000:223000, 4], zs=X_S[222000:223000, 5], c='black')
ax.scatter(xs=X_S[270000:270500, 3], ys=X_S[270000:270500, 4], zs=X_S[270000:270500, 5], c='magenta')
ax.scatter(xs=X_S[310000:310500, 3], ys=X_S[310000:310500, 4], zs=X_S[310000:310500, 5], c='brown')
#ax.set_xlim(-10,+10)
#ax.set_ylim(-10,+10)
#ax.set_zlim(-10,+10)
ax.set_title('Gyroscope Hand(Wrist)')
ax.legend(['Walking', 'Jogging', 'Standing', 'Upstair', 'Downstair', 'Sitting', 'Sitting_Car', 'Cycling'])
plt.show()


#Gyroscope Pocket
fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X_S[2000:2500, 9], ys=X_S[2000:2500, 10], zs=X_S[2000:2500, 11], c='red', alpha=0.4)
ax.scatter(xs=X_S[46000:46100, 9], ys=X_S[46000:46100, 10], zs=X_S[46000:46100, 11], c='green', alpha=0.7)
ax.scatter(xs=X_S[90000:91000, 9], ys=X_S[90000:91000, 10], zs=X_S[90000:91000, 11], c='blue')
ax.scatter(xs=X_S[140000:140500, 9], ys=X_S[140000:140500, 10], zs=X_S[140000:140500, 11], c='y')
ax.scatter(xs=X_S[180000:180500, 9], ys=X_S[180000:180500, 10], zs=X_S[180000:180500, 11], c='c')
ax.scatter(xs=X_S[222000:224000, 9], ys=X_S[222000:224000, 10], zs=X_S[222000:224000, 11], c='black')
ax.scatter(xs=X_S[270000:270500, 9], ys=X_S[270000:270500, 10], zs=X_S[270000:270500, 11], c='magenta')
ax.scatter(xs=X_S[310000:310500, 9], ys=X_S[310000:310500, 10], zs=X_S[310000:310500, 11], c='brown', alpha=0.5)
#ax.set_xlim(-10,+10)
#ax.set_ylim(-10,+10)
#ax.set_zlim(-10,+10)
ax.set_title('Gyroscope Pocket(Waist)')
ax.legend(['Walking', 'Jogging', 'Standing', 'Upstair', 'Downstair', 'Sitting', 'Sitting_Car', 'Cycling'])
plt.show()