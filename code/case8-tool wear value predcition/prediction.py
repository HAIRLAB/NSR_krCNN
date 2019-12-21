'''
------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by BeiT Zhou on 2018-08-30
#
# ------------------------------------------------------------------------
'''


import numpy as np
import pandas as pd
import scipy.io as sio 
import matplotlib.pyplot as plt
temp = sio.loadmat('data_get.mat')

series_signal = {}

signal1=temp['signal1'] # signal1: smcAC
signal2=temp['signal2'] # signal2: smcDC
signal3=temp['signal3'] # signal3: vib_table
signal4=temp['signal4'] # signal4: vib_spindle 
signal5=temp['signal5'] # signal5: AE table 
signal6=temp['signal6'] # signal6: AE spindle  

ranges = [[0, 13], [13, 26], [26, 40], [40, 47], [47, 56], [56, 66], [66, 86], [86, 98],
         [98, 104], [104, 112], [112, 117], [117, 130], [130, 137], [137, 143], [143, 145]]

index=range(ranges[1][0], ranges[1][1])
train1=np.delete(signal1,index,axis=0)
train2=np.delete(signal2,index,axis=0)
train3=np.delete(signal3,index,axis=0)
train4=np.delete(signal4,index,axis=0)
train5=np.delete(signal5,index,axis=0)
train6=np.delete(signal6,index,axis=0)

test1=signal1[index,:]
test2=signal2[index,:]
test3=signal3[index,:]
test4=signal4[index,:]
test5=signal5[index,:]
test6=signal6[index,:]


from sklearn import preprocessing
scaler1= preprocessing.StandardScaler().fit(train1)
train1=np.expand_dims(scaler1.transform(train1),axis=2)
test1=np.expand_dims(scaler1.transform(test1),axis=2)
scaler2= preprocessing.StandardScaler().fit(train2)
train2=np.expand_dims(scaler2.transform(train2),axis=2)
test2=np.expand_dims(scaler2.transform(test2),axis=2)
scaler3= preprocessing.StandardScaler().fit(train3)
train3=np.expand_dims(scaler3.transform(train3),axis=2)
test3=np.expand_dims(scaler3.transform(test3),axis=2)
scaler4= preprocessing.StandardScaler().fit(train4)
train4=np.expand_dims(scaler4.transform(train4),axis=2)
test4=np.expand_dims(scaler4.transform(test4),axis=2)
scaler5= preprocessing.StandardScaler().fit(train5)
train5=np.expand_dims(scaler5.transform(train5),axis=2)
test5=np.expand_dims(scaler5.transform(test5),axis=2)
scaler6= preprocessing.StandardScaler().fit(train6)
train6=np.expand_dims(scaler6.transform(train6),axis=2)
test6=np.expand_dims(scaler6.transform(test6),axis=2)

train=np.concatenate((train1,train2,train3,train4,train5,train6),axis=-1)
test=np.concatenate((test1,test2,test3,test4,test5,test6),axis=-1)
label = temp['vb'].T
MAX_LABEL = np.max(label)
scaled_label = label / MAX_LABEL


from sklearn.model_selection import train_test_split
signals_idx = ['signal1', 'signal2', 'signal3', 'signal4', 'signal5', 'signal6']
input_data1 = np.concatenate([series_signal['fft_' + key] for key in signals_idx], axis=-1)
input_data2 = np.concatenate([series_signal['scaled_' + key] for key in signals_idx], axis=-1)
input_data = input_data2

from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten
from keras.layers.pooling import MaxPool1D, AvgPool1D

def get_model(INPUT_SHAPE):
    """
    define a model
    INPUT_SHAPE: input dimension
    """
    initializer = 'he_normal'
    inputs = Input((INPUT_SHAPE[1], INPUT_SHAPE[2]))
    conv1 = Conv1D(16, 10, strides=5, activation='relu', kernel_initializer=initializer)(inputs)
    pool1 = AvgPool1D(pool_size=10)(conv1)
    conv2 = Conv1D(32, 5, strides=3, activation='relu', kernel_initializer=initializer)(pool1)
    pool2 = AvgPool1D(pool_size=5)(conv2)
    flat = Flatten()(pool2)
    dense1 = Dense(128, activation='relu', kernel_initializer=initializer)(flat)
    dense2 = Dense(20, activation='relu', kernel_initializer=initializer)(dense1)
    output = Dense(1)(dense2)
    model = Model(inputs, output)
    return model

from sklearn.metrics import mean_squared_error

ranges = [[0, 13], [13, 26], [26, 40], [40, 47], [47, 56], [56, 66], [66, 86], [86, 98],
         [98, 104], [104, 112], [112, 117], [117, 130], [130, 137], [137, 143], [143, 145]]
mses = []

i = 7
TEST_TIME = 1
best_model = None

for test in range(TEST_TIME):
    idx_range = range(ranges[i][0], ranges[i][1])
    train_input = np.delete(input_data, idx_range, axis=0)
    train_label = np.delete(label, idx_range, axis=0)
    val_input = input_data[idx_range]
    val_label = label[idx_range]
    model = get_model(input_data.shape)
    optimizer = optimizers.adam(lr=0.0001, decay=1e-5)
    model.compile(loss='mse', optimizer=optimizer)
    _ = model.fit(
        train_input, train_label, epochs=400, verbose=0,
        validation_data=(val_input, val_label)
    )
    mse = mean_squared_error(val_label, model.predict(val_input))
    print('TEST %d MSE: %f' % (test + 1, mse))



