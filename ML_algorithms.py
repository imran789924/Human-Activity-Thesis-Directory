#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:10:56 2020

@author: imran
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Final_Dataset_V3.csv')
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]



from scipy import stats

def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = np.empty(shape=(1,2400)), np.empty(shape=(1))
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        v = v.flatten()
        print(v.shape)
        labels = y.iloc[i: i + time_steps]
        Xs = np.vstack([Xs, v])
        #print(Xs[0].shape)
        ys = np.vstack([ys, stats.mode(labels)[0][0]])
    print(Xs.shape)
    return Xs, np.array(ys).reshape(-1, 1)



TIME_STEPS = 200
STEP = 40

X_1, y_1 = create_dataset(
    pd.DataFrame(X),
    y,
    TIME_STEPS,
    STEP
)


X_2 = X_1[1:8796]
y_2 = y_1[1:8796]


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_2 = le.fit_transform(y_2)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=.25, random_state=42, shuffle=True, stratify=y_2)


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_train_1 = pca.fit_transform(X_train)
X_test_1 = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_




from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train_1, y_train.ravel())


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train_1, y_train.ravel())

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500, random_state=0)
classifier.fit(X_train_1, y_train.ravel())

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train_1, y_train.ravel())




# Predicting the Test set results
y_pred = classifier.predict(X_test_1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))