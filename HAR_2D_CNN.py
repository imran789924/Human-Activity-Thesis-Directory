# Importing Libraries
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Importing Dataset
dataset = pd.read_csv("Final_Dataset_V3.csv")
#dataset = dataset.drop("Unnamed: 0", axis = 1)

# Label Encoding
label = LabelEncoder()
dataset["label"] = label.fit_transform(dataset["Activity"])

'''# Rebuilding Dataset
dataset = dataset.iloc[:, 1:15]'''

# Partitioning Data into Input-Output
X = dataset[["h_ac_X", "h_ac_Y", "h_ac_Z", "h_gy_X", "h_gy_Y", "h_gy_Z",
             "p_ac_X", "p_ac_Y", "p_ac_Z", "p_gy_X", "p_gy_Y", "p_gy_Z"]]

y = dataset["label"]

scaled_X = pd.DataFrame(data = X, columns = ["h_ac_X", "h_ac_Y", "h_ac_Z", 
"h_gy_X", "h_gy_Y", "h_gy_Z","p_ac_X", "p_ac_Y", "p_ac_Z", "p_gy_X", "p_gy_Y", "p_gy_Z"])

scaled_X['label'] = y.values


# Frame Preparation
import scipy.stats as stats

Fs = 20
frame_size = Fs*4 
hop_size = Fs*2 

def get_frames(dataset, frame_size, hop_size):

    N_FEATURES = 12

    frames = []
    labels = []
    for i in range(0, len(dataset) - frame_size, hop_size):
        h_ac_X = dataset["h_ac_X"].values[i: i + frame_size]
        h_ac_Y = dataset["h_ac_Y"].values[i: i + frame_size]
        h_ac_Z = dataset["h_ac_Z"].values[i: i + frame_size]
        h_gy_X = dataset["h_gy_X"].values[i: i + frame_size]
        h_gy_Y = dataset["h_gy_Y"].values[i: i + frame_size]
        h_gy_Z = dataset["h_gy_Z"].values[i: i + frame_size]
        p_ac_X = dataset["p_ac_X"].values[i: i + frame_size]
        p_ac_Y = dataset["p_ac_Y"].values[i: i + frame_size]
        p_ac_Z = dataset["p_ac_Z"].values[i: i + frame_size]
        p_gy_X = dataset["p_gy_X"].values[i: i + frame_size]
        p_gy_Y = dataset["p_gy_Y"].values[i: i + frame_size]
        p_gy_Z = dataset["p_gy_Z"].values[i: i + frame_size]

        
        # Retrieve the most often used label in this segment
        label = stats.mode(dataset['label'][i: i + frame_size])[0][0]
        frames.append([h_ac_X, h_ac_Y, h_ac_Z, 
        h_gy_X, h_gy_Y, h_gy_Z, p_ac_X, p_ac_Y, p_ac_Z, p_gy_X, p_gy_Y, p_gy_Z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)


# Train-Test-Spliting and Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 0, stratify = y)

temp_df = pd.DataFrame()
for i in range(X_train.shape[0]):
    temp_df = pd.concat([temp_df, pd.DataFrame(X_train[i], index = None)], axis = 0, ignore_index = True)
    
scaler = StandardScaler()
scaler = scaler.fit(temp_df)

for i in range(X_train.shape[0]):
    X_train[i] = scaler.transform(X_train[i])
    
for i in range(X_test.shape[0]):
    X_test[i] = scaler.transform(X_test[i])


# Reshaping to get 3D Data
X_train = X_train.reshape(7038, 80, 12, 1)
X_test = X_test.reshape(1760, 80, 12, 1)


# Implementing 2D CNN Model
model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = (X_train[0].shape)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(8, activation='softmax'))

model.compile(optimizer=Adam(lr = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 64, validation_data= (X_test, y_test), verbose=1)

model.evaluate(X_test, y_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))