# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:21:24 2018

@author: 马小跳
"""
import numpy as np
from copy import deepcopy
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.merge import concatenate
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from sklearn.cross_validation import train_test_split
import scipy.io as sio

label=np.zeros((1854,1))
label[:-662]=0
label[-662:]=1

data=np.concatenate((data1_vib,data2_vib,data3_vib,data4_vib),axis=0)
label=to_categorical(label,num_classes=2)

acc2=[]
nsr=np.asarray(list(range(50)))*10

def wgn(x, snr):
    xpower = np.sum(x**2)/len(x)
    npower = xpower * snr
    return np.sqrt(npower)
for ratio in nsr:
    print ('the ratio is',ratio)
    data2=deepcopy(data)
    for index in range(data2.shape[0]):
        data2[index,:]=(data2[index,:].reshape(1,-1)+wgn(data2[index,:], 10/100)*np.random.randn(1,data2.shape[1]))
    
    data2=np.expand_dims(data2,axis=2)

    train_data,test_data,train_label,test_label=train_test_split(data2,label,test_size=0.2,random_state=40)
    
    visible1=Input(shape=(6000,1))
    conv11=Conv1D (64,kernel_size=100,strides=100,activation='relu')(visible1)
    conv12=Conv1D (128,kernel_size=2,strides=2,activation='relu')(conv11)
    pool11=MaxPool1D(pool_size=2)(conv12)
    flat11=Flatten(name='flatten1')(pool11)
    hidden1=Dense(100,activation='relu')(flat11)
    output1=Dense(2,activation='softmax',name='result')(hidden1)
    model=Model(inputs=[visible1],outputs=[output1])
    model.summary()
    optimizer = optimizers.adam(lr=0.001)
    model.compile(
         loss='categorical_crossentropy',
         optimizer=optimizer,
         metrics=['accuracy'])
    hist=model.fit([train_data],[train_label], epochs=50,validation_data=(test_data,test_label))
    acc2.append(hist.history['val_acc'] [-1])
sio.savemat('acc2.mat',{'nsr':nsr, 'acc2':acc2})

'''plot acc curve'''
acc4=sio.loadmat('F:\\江南大学齿轮箱\\齿轮箱数据\\齿轮箱数据\\code\四分类\\acc4.mat')['acc4'].T
acc2=sio.loadmat('F:\\江南大学齿轮箱\\齿轮箱数据\\齿轮箱数据\\code\二分类\\acc2.mat')['acc2'].T
nsr=sio.loadmat('F:\\江南大学齿轮箱\\齿轮箱数据\\齿轮箱数据\\code\二分类\\acc2.mat')['nsr'].T

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
plt.ylim(50,105)
plt.xlim(0,500)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
#plt.axhline(y=90,c='k',linestyle='--',linewidth=2)
plt.text(202, 75, r'Noise ratio = 200%', fontdict=font1)    

plt.vlines(200,0,105,linestyle='--',linewidth=2)
plt.fill_between(nsr[:np.where(np.asarray(acc4)<0.97)[0][0],0], 105, color='yellow', alpha=0.2,label='Accuracy>97%')
plt.fill_between(nsr[np.where(np.asarray(acc4)<0.97)[0][0]:,0], 105, color='blue', alpha=0.03,label='Accuracy<97%')

plt.scatter(nsr,np.asarray(acc2[:,0])*100,color='', marker='o', edgecolors='green')
plt.plot(nsr,np.asarray(acc2[:,0])*100,linewidth=3,c='green',label='Two-way classification')

plt.plot(nsr,np.asarray(acc4[:])*100,linewidth=3,c='blueviolet',label='Four-way classification')
plt.scatter(nsr,np.asarray(acc4[:])*100,color='', marker='o', edgecolors='blueviolet')

#frame.set_facecolor('white') 
plt.legend(loc='lower right',prop=font2,facecolor ='white')

plt.savefig('noise acc.jpg',dpi=1200)  


#'''
#plot some noisey figures 0.50%; 200%; 500%; 
#'''
#
#def wgn(x, snr):
#    xpower = np.sum(x**2)/len(x)
#    npower = xpower * snr
#    return np.sqrt(npower)
#
#font1 = {'family' : 'Times New Roman',
#'weight' : 'normal',
#'size'   : 13,
#}
#font2 = {'family' : 'Times New Roman',
#'weight' : 'normal',
#'size'   : 14,
#}
#
#nsr=50
#noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
#raw_data=data[10,:]
#plt.plot(noise_data,c='orange',label='data with 50% noise')
#plt.plot(raw_data,c='k',label='raw data')
#plt.xlabel('Measurements',fontsize=15)
#plt.ylabel('Magnitude',fontsize=15)
#plt.xlim(0,6000)
#plt.ylim(-1,1)
#plt.tick_params(direction='in',width=1.5,length=4)
#plt.yticks(fontproperties = 'Times New Roman', size = 12)
#plt.xticks(fontproperties = 'Times New Roman', size = 12)
#plt.grid(linestyle='--')
#plt.legend(loc='upper right',prop=font2,facecolor ='white')
#plt.savefig('50% noise data.jpg',dpi=1200)
#
#
#nsr=200
#noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
#raw_data=data[10,:]
#plt.plot(noise_data,c='orange',label='data with 200% noise')
#plt.plot(raw_data,c='k',label='raw data')
#plt.xlabel('Measurements',fontsize=15)
#plt.ylabel('Magnitude',fontsize=15)
#plt.xlim(0,6000)
#plt.ylim(-1,1)
#plt.tick_params(direction='in',width=1.5,length=4)
#plt.yticks(fontproperties = 'Times New Roman', size = 12)
#plt.xticks(fontproperties = 'Times New Roman', size = 12)
#plt.grid(linestyle='--')
#plt.legend(loc='upper right',prop=font2,facecolor ='white')
#plt.savefig('200% noise data.jpg',dpi=1200)
#
#
#nsr=500
#noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
#raw_data=data[10,:]
#plt.plot(noise_data,c='orange',label='data with 500% noise')
#plt.plot(raw_data,c='k',label='raw data')
#plt.xlabel('Measurements',fontsize=15)
#plt.ylabel('Magnitude',fontsize=15)
#plt.xlim(0,6000)
#plt.ylim(-1,1)
#plt.tick_params(direction='in',width=1.5,length=4)
#plt.yticks(fontproperties = 'Times New Roman', size = 12)
#plt.xticks(fontproperties = 'Times New Roman', size = 12)
#plt.grid(linestyle='--')
#plt.legend(loc='upper right',prop=font2,facecolor ='white')
#plt.savefig('500% noise data.jpg',dpi=1200)













