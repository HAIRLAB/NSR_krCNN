
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.utils import to_categorical

from keras.layers import Input, Dense, Conv1D
from keras.layers import Flatten, Dropout, MaxPool1D
from keras.optimizers import SGD,Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import scipy.io as sio

data=sio.loadmat('data.mat')['data']
labels=sio.loadmat('data.mat')['labels']


def build_model():
    """
    build cnn model for training
    """
    inputs = Input((6000, 1))
    x = Conv1D(8, kernel_size=100, strides=100, activation='relu')(inputs)
    x = MaxPool1D(2)(x)
    x = Conv1D(8, kernel_size=2, strides=2, activation='relu')(x)
    x = MaxPool1D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    opt = Adam(lr=0.01, decay=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
  

model = build_model()

np.random.seed(20)

# start cnn training
model.fit(data, labels, validation_split=0.2, epochs=50, batch_size=20, verbose=2)

