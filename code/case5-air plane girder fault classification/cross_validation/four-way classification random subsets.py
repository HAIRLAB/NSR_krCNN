# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:28:20 2018

@author: 马小跳
"""

import numpy as np
import scipy.io as sio
temp=sio.loadmat('data_all.mat')
temp=temp['data']

noscrew=temp[:,:15,:]
screw2=temp[:,15:30,:]
screw4=temp[:,30:45,:]
screw6=temp[:,45:60,:]

data1=np.reshape(np.asarray(noscrew),(10000,300)).T
data2=np.reshape(np.asarray(screw2),(10000,300)).T
data3=np.reshape(np.asarray(screw4),(10000,300)).T
data4=np.reshape(np.asarray(screw6),(10000,300)).T


data=np.concatenate((data1,data2,data3,data4))
data=np.expand_dims(data,axis=2)
label=np.zeros((1200,1))
label[:300]=0
label[300:600]=1
label[600:900]=2
label[900:1200]=3

from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers
from sklearn.cross_validation import train_test_split

label=to_categorical(label)
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.1,random_state=40)


input=Input(shape=(10000,1))
conv1=Conv1D(128,kernel_size=1000,strides=100,activation='relu')(input)
pool1=MaxPool1D(pool_size=2)(conv1)
conv2=Conv1D(128,kernel_size=2,activation='relu')(pool1)
pool2=MaxPool1D(pool_size=2)(conv2)

flat1=Flatten(name='flatten')(pool2)

dense2=Dense(500,activation='relu')(flat1)
#dense3=Dense(100,activation='relu')(dense2)
#dense4=Dense(50,activation='relu')(dense2)

output=Dense(4,activation='softmax',name='result')(dense2)
model=Model(inputs=input,outputs=output)


optimizer = optimizers.adam(lr=0.001)
model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train, epochs=100,validation_data=(x_test,y_test))
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('result').output)
predict = dense1_layer_model.predict(x_test)

model.save('girder_position.h5')


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
                     index = ['No Screw','Screw no.2','Screw no.4','Screw no.6'],
                     columns = ['No Screw','Screw no.2','Screw no.4','Screw no.6'])
plt.gca().yaxis.set_tick_params(rotation=0)
sns.heatmap(cm_df, annot=True, fmt='d',cmap='Blues')
plt.gca().yaxis.set_tick_params(rotation=90)

plt.gca().xaxis.tick_top()
plt.xlabel('False Label')
plt.ylabel('True Label')

plt.savefig('girder_position.jpg',dpi=1200)

