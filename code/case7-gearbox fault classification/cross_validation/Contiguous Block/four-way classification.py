# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:21:24 2018

@author: 马小跳
"""
import numpy as np
label1=np.zeros((train_input1.shape[0],1))
label2=np.ones((train_input2.shape[0],1))*1
label3=np.ones((train_input3.shape[0],1))*2
label4=np.ones((train_input4.shape[0],1))*3
train_label=np.concatenate((label1,label2,label3,label4))

label1=np.zeros((test_input1.shape[0],1))
label2=np.ones((test_input2.shape[0],1))*1
label3=np.ones((test_input3.shape[0],1))*2
label4=np.ones((test_input4.shape[0],1))*3
test_label=np.concatenate((label1,label2,label3,label4))

train_data=np.concatenate((train_input1,train_input2,train_input3,train_input4),axis=0)
test_data=np.concatenate((test_input1,test_input2,test_input3,test_input4),axis=0)
train_data=np.expand_dims(train_data,axis=2)
test_data=np.expand_dims(test_data,axis=2)

from keras.utils import to_categorical
#label=to_categorical(label,num_classes=4)
train_label=to_categorical(train_label,num_classes=4)
test_label=to_categorical(test_label,num_classes=4)


from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


visible1=Input(shape=(6000,1))
conv11=Conv1D (64,kernel_size=100,strides=100,activation='relu')(visible1)
conv12=Conv1D (128,kernel_size=2,strides=2,activation='relu')(conv11)
pool11=MaxPool1D(pool_size=2)(conv12)
flat11=Flatten(name='flatten1')(pool11)

hidden1=Dense(100,activation='relu')(flat11)
 #hidden2=Dense(100,activation='relu')(hidden1)
output1=Dense(4,activation='softmax',name='result')(hidden1)


model=Model(inputs=[visible1],outputs=[output1])
model.summary()
optimizer = optimizers.adam(lr=0.001)
model.compile(
     loss='categorical_crossentropy',
     optimizer=optimizer,
     metrics=['accuracy'])
history=model.fit([train_data],[train_label], epochs=1000,validation_data=(test_data,test_label))
#model.save('vib_6000_split_2.h5')
#
#predict=model.predict(test_data)
###calculate recall precision
##for i in range(test.shape[1]):
##    for j in range(test.shape[0]):
##        if test[j,i]>=0.5:
##            test[j,i]=1
##        if test[j,i]<0.5:
##            test[j,i]=0
##for j in range(test.shape[0]):          
##    for i in range(test.shape[1]):
##        if test[j,i]==1:
##            test[j,:]=np.ones((1,test.shape[1]))*i
##            break
##test=test[:,0]
##
##for j in range(test_label.shape[0]):          
##    for i in range(test_label.shape[1]):
##        if test_label[j,i]==1:
##            test_label[j,:]=np.ones((1,test_label.shape[1]))*i
##            break
##test_label=test_label[:,0]
##
##TP = np.sum(np.logical_and(np.equal(test_label, 0), np.equal(test, 0)))
#
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_curve
#from sklearn.metrics import auc
#import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns
#import scipy.io as sio
#import numpy as np
#
#for i in range(predict.shape[0]):
#    for j in range(predict.shape[1]):
#        if predict[i,j]<0.5:
#            predict[i, j]=0
#        elif predict[i,j]>=0.5:
#            predict[i,j]=1
#test_label=test_label.argmax(1)
#predict=predict.argmax(1)               #one-hot 返回原始
#cm=confusion_matrix(test_label,predict) #求模糊矩阵
#print (cm)
#
#cm_df = pd.DataFrame(cm,
#                     index = ['','','',''],
#                     columns = ['','','',''])
#plt.gca().yaxis.set_tick_params(rotation=0)
#sns.heatmap(cm_df, annot=True, fmt='d',cmap='Blues')
#plt.gca().yaxis.set_tick_params(rotation=90)
#
#plt.gca().xaxis.tick_top()
#plt.xlabel('False Label')
#plt.ylabel('True Label')
#
#plt.savefig('gearbox4.jpg',dpi=1200)


