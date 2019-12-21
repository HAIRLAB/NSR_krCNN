# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:21:24 2018

@author: 马小跳
"""
import numpy as np
from copy import deepcopy
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.io as sio
sss=StratifiedShuffleSplit(test_size=0.2,n_splits=10,random_state=40)

label=np.zeros((1854,1))
label[0:715]=0
label[715:715+294]=1
label[715+294:715+294+183]=2
label[715+294+183:715+294+183+662]=3
data=np.concatenate((data1_vib,data2_vib,data3_vib,data4_vib),axis=0)
acc=[]
nsr=np.asarray(list(range(50)))*10
def wgn(x, snr):
    xpower = np.sum(x**2)/len(x)
    npower = xpower * snr
    return np.sqrt(npower)

for ratio in nsr:
    data2=deepcopy(data)
    for index in range(data2.shape[0]):
        data2[index,:]=(data2[index,:].reshape(1,-1)+wgn(data2[index,:], ratio/100)*np.random.randn(1,data2.shape[1]))
    data2=np.expand_dims(data2,axis=2)
    for train_index, test_index in sss.split(data2,label):
        train_data=data2[train_index]
        train_label=label[train_index]
        test_data=data2[test_index]
        test_label=label[test_index]
    train_label=to_categorical(train_label,num_classes=4)
    test_label=to_categorical(test_label,num_classes=4)
    
    visible1=Input(shape=(6000,1))
    conv11=Conv1D (64,kernel_size=100,strides=100,activation='relu')(visible1)
    conv12=Conv1D (128,kernel_size=2,strides=2,activation='relu')(conv11)
    pool11=MaxPool1D(pool_size=2)(conv12)
    flat11=Flatten(name='flatten1')(pool11)
    
    hidden1=Dense(100,activation='relu')(flat11)
    output1=Dense(4,activation='softmax',name='result')(hidden1)
    
    model=Model(inputs=[visible1],outputs=[output1])
    model.summary()
    optimizer = optimizers.adam(lr=0.001)
    model.compile(
         loss='categorical_crossentropy',
         optimizer=optimizer,
         metrics=['accuracy'])
    hist=model.fit([train_data],[train_label], epochs=50,validation_data=(test_data,test_label))
    acc.append(hist.history['val_acc'] [-1])
plt.plot(nsr,acc)
sio.savemat('acc4.mat',{'nsr':nsr, 'acc4':acc})

