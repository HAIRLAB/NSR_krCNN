
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
from copy import deepcopy
from sklearn.model_selection import StratifiedShuffleSplit
sss=StratifiedShuffleSplit(test_size=0.2,n_splits=10,random_state=40)
np.random.seed(20)

data=sio.loadmat('data.mat')['data']
labels=sio.loadmat('data.mat')['labels']

def wgn(x, snr):
    xpower = np.sum(x**2)/len(x)
    npower = xpower * snr
    return np.sqrt(npower)
nsr=np.asarray(list(range(61)))*10
acc=[]
"""
build cnn model for training
"""
for ratio in nsr:
    data2=deepcopy(data)
    data2=data2.reshape(-1,6000)
    np.random.seed(20)
    for index in range(data2.shape[0]):
        data2[index,:]=(data2[index,:].reshape(1,-1)+wgn(data2[index,:], ratio/100)*np.random.randn(1,data2.shape[1]))
    data2=np.expand_dims(data2,axis=2)
    for train_index, test_index in sss.split(data2,labels):
        train_data=data2[train_index]
        train_label=labels[train_index]
        test_data=data2[test_index]
        test_label=labels[test_index]
         
    inputs = Input((6000, 1))
    x = Conv1D(8, kernel_size=100, strides=100, activation='relu')(inputs)
    x = MaxPool1D(2)(x)
    x = Conv1D(8, kernel_size=2, strides=1, activation='relu')(x)
    x = MaxPool1D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    opt = Adam(lr=0.001, decay=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    # start cnn training
    hist=model.fit(train_data, train_label, validation_data=(test_data,test_label), 
              epochs=50, batch_size=20, verbose=2)
    acc.append(hist.history['val_acc'][-1])
    del hist
    del model
sio.savemat('acc.mat',{'nsr':nsr, 'acc':acc})

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

plt.xlabel('Noise ratio (%)',fontsize=15)
plt.ylabel('Accuracy (%)',fontsize=15)
plt.ylim(0,105)
plt.xlim(0,600)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
#plt.axhline(y=90,c='k',linestyle='--',linewidth=2)
plt.text(140, 42, r'Noise ratio = 120%', fontdict=font2)    

plt.vlines(120,0,105,linestyle='--',linewidth=2)
plt.fill_between(nsr[:np.where(np.asarray(acc)<0.990)[0][0]], 105, color='yellow', alpha=0.2,label='Accuracy=100%')
plt.fill_between(nsr[np.where(np.asarray(acc)<0.990)[0][0]:], 105, color='blue', alpha=0.03,label='Accuracy<100%')

plt.scatter(nsr,np.asarray(acc[:])*100,color='', marker='o', edgecolors='green')
plt.plot(nsr,np.asarray(acc[:])*100,linewidth=3,c='green',label='Two-way classification')

#frame.set_facecolor('white') 
plt.legend(loc='lower right',prop=font2,facecolor ='white')
plt.savefig('noise acc.jpg',dpi=1200)



'''
plot some noisey figures 10%; 200%; 600%; 
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

nsr=10
noise_data=(data[17,:].reshape(1,-1)+wgn(data[17,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[17,:]
plt.plot(noise_data,c='orange',label='data with 10% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,6000)
plt.ylim(-10,10)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('10% noise data.jpg',dpi=1200)


nsr=200
noise_data=(data[17,:].reshape(1,-1)+wgn(data[17,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[17,:]
plt.plot(noise_data,c='orange',label='data with 200% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,6000)
plt.ylim(-20,20)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('200% noise data.jpg',dpi=1200)

nsr=600
noise_data=(data[17,:].reshape(1,-1)+wgn(data[17,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[17,:]
plt.plot(noise_data,c='orange',label='data with 600% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,6000)
plt.ylim(-40,40)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('600% noise data.jpg',dpi=1200)


