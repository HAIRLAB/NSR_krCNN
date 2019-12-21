# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:06:51 2017

"""
import os
#
## GPU assigned
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'
#
#import tensorflow as tf
#
#config = tf.ConfigProto()
#
#config.gpu_options.allow_growth = True
## config.gpu_options.per_process_gpu_memory_fraction = 0.2
#
#config.allow_soft_placement = True
#sess = tf.Session(config=config)

import numpy as np
import scipy.io as sio
result=sio.loadmat('data.mat')
data=result['data']
#data=[]
#for i in range(489):
#    data.append(temp[0][i])
#del data[24],data[25],data[26]
#temp_data=np.concatenate((data[0],data[1]))
#for i in range(2,495):
#    temp_data=np.concatenate((temp_data,data[i]))
label=result['label']
#data=(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))*2-1
data=data[:,::10]
#data=np.expand_dims(data,axis=2)
from keras.utils import to_categorical
label=to_categorical(label)
from sklearn.model_selection import StratifiedShuffleSplit
#sss=StratifiedShuffleSplit(test_size=0.2,n_splits=10,random_state=40)
#for train_index,test_index in sss.split(data,label):
#    x_train=data[train_index]
#    y_train=label[train_index]
#    x_test=data[test_index]
#    y_test=label[test_index]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2,random_state=40)

from sklearn import preprocessing
scaler= preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_train=np.expand_dims(x_train,axis=2)
x_test=np.expand_dims(x_test,axis=2)




# print x_train.shape,x_test.shape,y_train.shape,y_test.shape
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers
input=Input(shape=(41000,1))
conv1=Conv1D(64,kernel_size=2000,strides=200,activation='relu')(input)
pool1=MaxPool1D(pool_size=2)(conv1)
#conv2=Conv1D(128,kernel_size=2,activation='relu')(pool1)
##pool2=MaxPool1D(pool_size=2)(conv2)
flat1=Flatten(name='flatten')(pool1)

dense2=Dense(500,activation='relu')(flat1)
#dense3=Dense(100,activation='relu')(dense2)
#dense4=Dense(50,activation='relu')(dense2)

output=Dense(4,activation='softmax',name='result')(dense2)
model=Model(inputs=input,outputs=output)


optimizer = optimizers.adam(lr=0.0001)
model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train, epochs=1000,validation_data=(x_test,y_test),verbose=2)

dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('result').output)
predict = dense1_layer_model.predict(x_test)

model.save('wuxifenlei.h5')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io as sio
import numpy as np

for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        if predict[i,j]<0.5:
            predict[i, j]=0
        elif predict[i,j]>=0.5:
            predict[i,j]=1
test_label=y_test.argmax(1)
predict=predict.argmax(1)               #one-hot 返回原始
cm=confusion_matrix(test_label,predict) #求模糊矩阵
print (cm)

cm_df = pd.DataFrame(cm,
                     index = ['Processing 1','Processing 3','Processing 3','Processing 4'],
                     columns = ['Processing 1','Processing 3','Processing 3','Processing 4'])
plt.gca().yaxis.set_tick_params(rotation=0)
sns.heatmap(cm_df, annot=True, fmt='d',cmap='Blues')
plt.gca().yaxis.set_tick_params(rotation=90)

plt.gca().xaxis.tick_top()
plt.xlabel('False Label')
plt.ylabel('True Label')

plt.savefig('wuxi.jpg',dpi=1200)




