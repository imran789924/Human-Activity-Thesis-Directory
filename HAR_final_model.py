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



'''
train_pct_index = int(0.8 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
'''



from scipy import stats


def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        print(v.shape)
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)



TIME_STEPS = 200
STEP = 40

X, y = create_dataset(
    pd.DataFrame(X),
    y,
    TIME_STEPS,
    STEP
)


'''X_test, y_test = create_dataset(
    X_test,
    y_test,
    TIME_STEPS,
    STEP
)'''
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0, shuffle=True, stratify=y)

#print(X_train.shape, y_train.shape)



df1 = pd.DataFrame()
for i in range (X_train.shape[0]):
    df1 = pd.concat([df1, pd.DataFrame(X_train[i], index=None)], axis=0, ignore_index=True)
scaler = scaler.fit(df1)
for i in range (X_train.shape[0]):
    X_train[i] = scaler.transform(X_train[i])
for i in range(X_test.shape[0]):
    X_test[i] = scaler.transform(X_test[i])



from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)



import keras



model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=200,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)


history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=100,
    validation_split=0.1,
    shuffle=False
)


model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

#labels=['Walking', 'Jogging', 'Standing', 'Upstair', 'Downstair', 'Sitting', 'Car', 'Cycling']

y_pred = np.float64(y_pred)
rev_y_pred = enc.inverse_transform(y_pred)
rev_y_test = enc.inverse_transform(y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rev_y_test, rev_y_pred)


cmtx = pd.DataFrame(
    cm, 
    index=['true:Walking', 'true:Jogging', 'true:Standing', 'true:Upstair', 'true:Downstair', 'true:Sitting', 'true:Sitting_Car', 'true:Cycling'], 
    columns=['pred:Walking', 'pred:Jogging', 'pred:Standing', 'pred:Upstair', 'pred:Downstair', 'pred:Sitting', 'pred:Sitting_Car', 'pred:Cycling']
)


from sklearn.metrics import classification_report
print(classification_report(rev_y_test, rev_y_pred))