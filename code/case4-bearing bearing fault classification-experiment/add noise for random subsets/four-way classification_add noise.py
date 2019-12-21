# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 08:07:27 2018

@author: 马小跳
"""

import numpy as np 
import csv
import matplotlib.pyplot as plt
import os
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers
from sklearn.cross_validation import train_test_split 
from copy import deepcopy
np.random.seed(40)

path='F:\\江南大学轴承\\数据'
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


#add noise 
def wgn(x, snr):
#    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower * snr
    return np.sqrt(npower)
nsr=np.asarray(list(range(51)))*4
acc=[]
label=to_categorical(label)

for ratio in nsr:
    print ('the ratio is', ratio)
    data2=deepcopy(data)
    for index in range(data2.shape[0]):
        data2[index,:]=(data2[index,:].reshape(1,-1)+wgn(data2[index,:], ratio/100)*np.random.randn(1,data2.shape[1]))

    
    x_train,x_test,y_train,y_test=train_test_split(data2,label,test_size=0.1,random_state=10)
    x_train=np.expand_dims(x_train,axis=2)
    x_test=np.expand_dims(x_test,axis=2)
    
    
    input=Input(shape=(10000,1))
    conv1=Conv1D(64,kernel_size=200,strides=50,activation='tanh')(input)
    pool1=MaxPool1D(pool_size=4)(conv1)
    conv2=Conv1D(64,kernel_size=2,strides=1,activation='tanh')(pool1)
    pool2=MaxPool1D(pool_size=4)(conv2)
    
    flat1=Flatten(name='flatten')(pool2)
    
    dense2=Dense(500,activation='relu')(flat1)
    
    output=Dense(4,activation='softmax',name='result')(dense2)
    model=Model(inputs=input,outputs=output)
    
    
    optimizer = optimizers.adam(lr=0.005)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary()
    hist=model.fit(x_train,y_train, epochs=50,validation_data=(x_test,y_test))
    acc.append(hist.history['val_acc'][-1])
    del data2
    del x_train
    del x_test
    del model 
    del hist

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

plt.xlabel('Noise ratio (%)',fontsize=15)
plt.ylabel('Accuracy (%)',fontsize=15)
plt.ylim(60,105)
plt.xlim(0,200)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')

plt.text(25, 72, r'Noise ratio = 20%', fontdict=font1)    
plt.vlines(20,0,105,linestyle='--',linewidth=2)
plt.fill_between(nsr[:np.where(np.asarray(acc)<0.97)[0][0]], 105, color='yellow', alpha=0.2,label='Accuracy>97%')
plt.fill_between(nsr[np.where(np.asarray(acc)<0.97)[0][0]:], 105, color='blue', alpha=0.03,label='Accuracy<97%')


plt.plot(nsr,np.asarray(acc)*100,linewidth=3,c='k',label='Four-way classification')
plt.scatter(nsr,np.asarray(acc)*100,color='', marker='o', edgecolors='k')
plt.legend(loc='lower right',prop=font2,facecolor ='white')

plt.savefig('noise acc.jpg',dpi=1200)


'''
plot some noisey figures 20%; 100%; 200%; 
'''
def wgn(x, snr):
    xpower = np.sum(x**2)/len(x)
    npower = xpower * snr
    return np.sqrt(npower)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

nsr=20
noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[10,:]
plt.plot(noise_data,c='orange',label='data with 20% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,10000)
plt.ylim(-10,10)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('20% noise data.jpg',dpi=1200)


nsr=100
noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[10,:]
plt.plot(noise_data,c='orange',label='data with 100% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,10000)
plt.ylim(-10,10)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('100% noise data.jpg',dpi=1200)

nsr=200
noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[10,:]
plt.plot(noise_data,c='orange',label='data with 200% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,10000)
plt.ylim(-10,10)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('200% noise data.jpg',dpi=1200)       
        