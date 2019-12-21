# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 08:07:27 2018

@author: 马小跳
"""

import numpy as np 
import csv
import matplotlib.pyplot as plt
import os 
np.random.seed(40)
path='path'
files=os.listdir(path)
temp_data=[]
temp_data_all=[]
for item in files:
   if 'csv' in item:
       root=os.path.join(path,item)
       csv_reader = csv.reader(open(root, encoding='utf-8'))
       for row in csv_reader:
           temp_data.append(row[0])
       temp_data=np.asarray(temp_data)
       temp_data=temp_data.astype(np.float64)
       if temp_data.shape[0]==500500:        
           temp_data_all.append(temp_data[:500000])
       else:
#           temp_data=np.reshape(temp_data,(3,500500))
           temp_data_all.append(temp_data[:1500000])
       temp_data=[]
data=np.concatenate((temp_data_all[0],temp_data_all[1]),axis=0)
for i in range(2,len(temp_data_all)):
    data=np.concatenate((data,temp_data_all[i]),axis=0)
    
    
data=np.reshape(data,(18,500000))
data=np.reshape(data,(900,10000))
#data1=2*(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))-1
#data1=np.expand_dims(data1,axis=2)
#normalize = lambda series: 2*(series - np.min(series)) / (np.max(series) - np.min(series))-1
#data=np.asarray(list(map(normalize, data)))

#label=[0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,3,3,3]
label0=np.zeros((150,1))
label1=np.ones((450,1))
label2=np.zeros((150,1))
label2[:,:]=2
label3=np.zeros((150,1))
label3[:,:]=3
label=np.concatenate((label0,label1,label2,label3))


from keras.utils import to_categorical
label=to_categorical(label)
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.1,random_state=10)


#mean_value=np.mean(x_train,axis=0)
#max_value=np.max(x_train,axis=0)
#min_value=np.min(x_train,axis=0)
#x_train=(x_train-mean_value)/(max_value-min_value)
#x_test=(x_test-mean_value)/(max_value-min_value)
#from sklearn import preprocessing
#scaler= preprocessing.StandardScaler().fit(x_train)
#train_data=scaler.transform(x_train)
#test_data=scaler.transform(x_test)

x_train=np.expand_dims(x_train,axis=2)
x_test=np.expand_dims(x_test,axis=2)


input=Input(shape=(10000,1))
conv1=Conv1D(32,kernel_size=1000,strides=50,activation='relu')(input)
pool1=MaxPool1D(pool_size=2)(conv1)
conv2=Conv1D(64,kernel_size=2,strides=1,activation='relu')(pool1)
pool2=MaxPool1D(pool_size=2)(conv2)
#conv3=Conv1D(64,kernel_size=2,activation='relu')(pool2)
#pool3=MaxPool1D(pool_size=2)(conv3)



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
history=model.fit(x_train,y_train, epochs=1000,validation_data=(x_test,y_test))
dense1_layer_model = Model(inputs=model.input,  
                           outputs=model.get_layer('result').output)
predict = dense1_layer_model.predict(x_test)



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
                     index = ['Inner race fault','Normal','Outer race fault','Rolling elements fault'],
                     columns = ['Inner race fault','Normal','Outer race fault','Rolling elements fault'])
plt.gca().yaxis.set_tick_params(rotation=0)
sns.heatmap(cm_df, annot=True, fmt='d',cmap='Blues')
plt.gca().yaxis.set_tick_params(rotation=90)

plt.gca().xaxis.tick_top()
plt.xlabel('False Label')
plt.ylabel('True Label')

plt.savefig('jiangnanbearing4.jpg',dpi=1200)








##calculate recall precision
#test=dense1_output1
#for i in range(test.shape[1]):
#    for j in range(test.shape[0]):
#        if test[j,i]>=0.5:
#            test[j,i]=1
#        if test[j,i]<0.5:
#            test[j,i]=0
#for j in range(test.shape[0]):          
#    for i in range(test.shape[1]):
#        if test[j,i]==1:
#            test[j,:]=np.ones((1,test.shape[1]))*i
#            break
#test=test[:,0]
#
#
#test_label=y_test
#for j in range(test_label.shape[0]):          
#    for i in range(test_label.shape[1]):
#        if test_label[j,i]==1:
#            test_label[j,:]=np.ones((1,test_label.shape[1]))*i
#            break
#test_label=test_label[:,0]
#
#for i in range(test.shape[0]):
#    if test[i]!=test_label[i]:
#        print (i)
#        
#np.sum(np.equal(test_label,3))        
        
        