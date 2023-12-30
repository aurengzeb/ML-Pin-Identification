# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

import numpy as np
# load the dataset
import pandas as pd
df = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/clean3.csv", header=None)
X = df[df.columns[1:207]]
y = df[df.columns[0:1]]
all_X = np.array(X, dtype=np.float)
all_y = np.array(y, dtype=np.float)
shape = all_X.shape
maxx = np.max(all_X)

##data = array(all_y)
##print(data)
### one hot encode
##encoded = to_categorical(data)
##print(encoded)
### invert encoding
##inverted = argmax(encoded[0])
##print(inverted)

# split into input (X) and output (y) variables
##X = dataset[:,0:209]
##y = dataset[:,209]
# define the keras model
model = Sequential()
model.add(Dense(2*shape[1]+1, input_dim=shape[1], activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(int((2*shape[1]+1)/2), activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(int((2*shape[1]+1)/3), activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(all_X, all_y, epochs=150, batch_size=10, verbose=1,validation_split=0.1)
# make class predictions with the model
#predictions = model.predict_classes(X)

##model = Sequential()
##model.add(Dense(12, input_dim=shape[1], activation='relu'))
##model.add(Dense(8, activation='relu'))
##model.add(Activation('softmax'))
#model.add(Dense(NUM_CLASSES=2, activation='softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
##model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Training...")
##model.fit(all_X, all_y, nb_epoch=10, batch_size=150, validation_split=0.1,verbose=2)

##print("Generating test predictions...")
##preds = model.predict_classes(X_test, verbose=0)
##
##def write_preds(preds, fname):
##    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)
##
##write_preds(preds, "keras-mlp.csv")