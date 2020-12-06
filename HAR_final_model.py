import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Final_Dataset_V1.csv')
X = df.iloc[:,3:16]
y = df.iloc[:, 1]



from sklearn.model_selection import train_test_split




#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,shuffle=False)

'''
train_pct_index = int(0.8 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
'''

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X)
#X_test = scaler.fit_transform(X_test)



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

X_train, y_train = create_dataset(
    pd.DataFrame(X_train),
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


#print(X_train.shape, y_train.shape)



from sklearn.preprocessing import OneHotEncoder


enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
#y_test = enc.transform(y_test)



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
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)


history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=100,
    validation_split=0.1,
    shuffle=False
)


#model.evaluate(X_test, y_test)