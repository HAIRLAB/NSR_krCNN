# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:00:17 2018

@author: 马小跳
"""

import numpy as np
import scipy.io as sio
temp=sio.loadmat('F:\\大梁数据\\raw data\\data_all')
temp=temp['data']
data=[]
for i in range(temp.shape[1]):
    for j in range(temp.shape[2]):
        data.append(temp[:,i,j])
data=np.asarray(data)  

data1=[]
data2=[]
data3=[]
data4=[]
data5=[]
for i in range(4):
    data1.append(data[300*i:300*i+60])
    data2.append(data[300*i+60:300*i+120])
    data3.append(data[300*i+120:300*i+180])
    data4.append(data[300*i+180:300*i+240])
    data5.append(data[300*i+240:300*i+300])
data1=np.reshape(np.asarray(data1),(240,10000))
data2=np.reshape(np.asarray(data2),(240,10000))
data3=np.reshape(np.asarray(data3),(240,10000))
data4=np.reshape(np.asarray(data4),(240,10000))
data5=np.reshape(np.asarray(data5),(240,10000))

data=np.concatenate((data1,data2,data3,data4,data5))

      
#data1=data[0:300,:]
label=np.zeros((1200,1))
label[:240]=0
label[240:480]=1
label[480:720]=2
label[720:960]=3
label[960:1200]=4

data=np.expand_dims(data,axis=2)
from keras.utils import to_categorical
label=to_categorical(label)
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers
from sklearn.cross_validation import train_test_split
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

output=Dense(5,activation='softmax',name='result')(dense2)
model=Model(inputs=input,outputs=output)


optimizer = optimizers.adam(lr=0.001)
model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train, epochs=1000,validation_data=(x_test,y_test))

dense1_layer_model = Model(inputs=model.input,  
                           outputs=model.get_layer('result').output)
predict = dense1_layer_model.predict(x_test)


model.save('girder_degree.h5')


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
                     index = ['No Damage','4 mm Damage','8 mm Damage','12 mm Damage','16 mm Damage'],
                     columns = ['No Damage','4 mm Damage','8 mm Damage','12 mm Damage','16 mm Damage'])
plt.gca().yaxis.set_tick_params(rotation=0)
sns.heatmap(cm_df, annot=True, fmt='d',cmap='Blues')
plt.gca().yaxis.set_tick_params(rotation=90)

plt.gca().xaxis.tick_top()
plt.xlabel('False Label')
plt.ylabel('True Label')

plt.savefig('girder_degree.jpg',dpi=1200)





