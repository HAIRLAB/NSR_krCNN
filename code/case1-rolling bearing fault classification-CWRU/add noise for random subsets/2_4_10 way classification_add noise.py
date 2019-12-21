'''
------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by GuiJ Ma on 2018-08-30
#
# ------------------------------------------------------------------------
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
def load_fault_data():   
    '''
    Load fault_data of bearing from a sample frequency of 12k
    including rolling element, inner race and outer race fault with 7-mil 
    14-mil,21-mil diameter, we use DE variable for working
    Reshape the raw data in a format of (samples,6000)
    '''
    
    path='12kDriveEnd'
    files=os.listdir(path)
    temp=[]
    label=[]
    for mat in files:
        if '28' not in mat:
            temp1=sio.loadmat(os.path.join(path,mat))
            for key in temp1.keys():
                if 'DE' in key:
                    temp.append(temp1[key][:120000])
                    if 'B' in mat:
                        if '07' in mat:                           
                            label.append([0]*20)
                        if '14' in mat:
                            label.append([1]*20)
                        if '21' in mat:
                            label.append([2]*20)
                    if 'IR' in mat:
                        if '07' in mat:                           
                            label.append([3]*20)
                        if '14' in mat:
                            label.append([4]*20)
                        if '21' in mat:
                            label.append([5]*20)
                    if 'OR' in mat:
                        if '07' in mat:                           
                            label.append([6]*20)
                        if '14' in mat:
                            label.append([7]*20)
                        if '21' in mat:
                            label.append([8]*20)
    temp=np.asarray(temp)
    data1=temp.reshape((-1,6000))
    label1=np.asarray(label)
    label1=label1.reshape((-1,1))
    return data1, label1

def load_normal_data():
    '''
    Load normal_data of bearing
    we use DE variable for working
    Reshape the raw data in a format of (samples,6000)
    '''    
    
    path='Normal_Baseline_Data'
    files=os.listdir(path)
    temp=[]
    label2=[]
    for mat in files:
        temp1=sio.loadmat(os.path.join(path,mat))
        for key in temp1.keys():
            if 'DE' in key:
                if 240000<len(temp1[key])<480000:
                    temp.append(temp1[key][:240000])
                if len(temp1[key])>480000:
                    temp.append(temp1[key][:480000])
    temp2=np.concatenate((temp[0],temp[1],temp[2],temp[3]))
    data2=temp2.reshape((-1,6000))
    label2=np.ones((data2.shape[0],1))*9
    return data2,label2
def concatenate_data():
    '''
    combine all data to be a set
    '''    
    
    data1,label1=load_fault_data()
    data2,label2=load_normal_data()
    data=np.concatenate((data1,data2))
    label=np.concatenate((label1,label2))
    return data,label
data,label=concatenate_data()

def wgn(x, snr):
#    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower * snr
    return np.sqrt(npower)


'''
plot some noisey figures 10%; 100%; 400%; 
'''


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

nsr=10
noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[10,:]
plt.plot(noise_data,c='orange',label='data with 10% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,6000)
plt.ylim(-0.8,0.8)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('10% noise data.jpg',dpi=1200)

nsr=100
noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[10,:]
plt.plot(noise_data,c='orange',label='data with 100% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,6000)
plt.ylim(-1,1)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('100% noise data.jpg',dpi=1200)

nsr=400
noise_data=(data[10,:].reshape(1,-1)+wgn(data[10,:], nsr/100)*np.random.randn(1,data.shape[1])).T
raw_data=data[10,:]
plt.plot(noise_data,c='orange',label='data with 400% noise')
plt.plot(raw_data,c='k',label='raw data')
plt.xlabel('Measurements',fontsize=15)
plt.ylabel('Magnitude',fontsize=15)
plt.xlim(0,6000)
plt.ylim(-2,2)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
plt.legend(loc='upper right',prop=font2,facecolor ='white')
plt.savefig('400% noise data.jpg',dpi=1200)

'''
ten-way classification
'''
accuracy10=[]
from keras.models import Model
from keras.layers import Dense,Flatten,Input
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras import layers
from keras import optimizers
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from keras import models
from keras.utils import to_categorical
nsr=np.asarray(list(range(0,50)))*10
for ratio in nsr:
    print (f'this is the {ratio}% noise')
    data = np.expand_dims(data, axis=2)
    label=to_categorical(label)
    for index in range(data.shape[0]):
        data[index,:,:]=(data[index,:].reshape(1,-1)+wgn(data[index,:], ratio/100)*np.random.randn(1,data.shape[1])).T
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1,random_state=40)
    model = Sequential()
    model.add(Conv1D(64, kernel_size=50,strides=50, activation='relu', input_shape=(6000, 1, )))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64,kernel_size=2,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(10, activation='softmax'))
    optimizer = optimizers.adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    model.summary()
    hist=model.fit(train_data, train_label, epochs=50,verbose=1,shuffle=True,validation_data=(test_data,test_label))
    predict=model.predict(test_data)
    data,label=concatenate_data()
    accuracy10.append(hist.history['val_acc'] [-1])
#plt.plot(nsr[:50],accuracy10[:50])
#plt.plot(nsr[:10],accuracy2[:10])
#plt.plot(nsr[:10],accuracy4[:10])
#
#plt.plot(a.T)
#plt.plot(data[0,:])
'''
end
'''
#--------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
'''
four-way classification
'''
'''add noise'''
#accuracy4=[]
#from keras.models import Model
#from keras.layers import Dense,Flatten,Input
#from keras.layers import Conv1D, MaxPooling1D
#from keras.optimizers import SGD
#from keras import layers
#from keras import optimizers
#from keras.models import Sequential
#from sklearn.cross_validation import train_test_split
#from keras import models
#from keras.utils import to_categorical
#nsr=np.asarray(list(range(0,50)))*10
#for ratio in nsr:
#    print (f'this is the {ratio}% noise')
#    data = np.expand_dims(data, axis=2)
#    for i in range(len(label)):
#        if label[i]==0:
#            label[i]=0
#        if label[i]==1:
#            label[i]=0
#        if label[i]==2:
#            label[i]=0
#        if label[i]==3:
#            label[i]=1
#        if label[i]==4:
#            label[i]=1
#        if label[i]==5:
#            label[i]=1
#        if label[i]==6:
#            label[i]=2
#        if label[i]==7:
#            label[i]=2  
#        if label[i]==8:
#            label[i]=2
#        if label[i]==9:
#            label[i]=3
#    label=to_categorical(label)
#    for index in range(data.shape[0]):
#        data[index,:,:]=(data[index,:].reshape(1,-1)+
#            wgn(data[index,:],ratio/100)*np.random.randn(1,data.shape[1])).T
#    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1,random_state=40)
#    model = Sequential()
#    model.add(Conv1D(64, kernel_size=50,strides=50, activation='relu', input_shape=(6000, 1, )))
#    model.add(MaxPooling1D(pool_size=2))
#    model.add(Conv1D(64,kernel_size=2,activation='relu'))
#    model.add(MaxPooling1D(pool_size=2))
#    model.add(Flatten(name='flatten'))
#    model.add(Dense(4, activation='softmax'))
#    optimizer = optimizers.adam(lr=0.001,decay=1e-5)
#    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
#    model.summary()
#    hist=model.fit(train_data, train_label, epochs=100,verbose=1,shuffle=True,validation_data=(test_data,test_label))
#    predict=model.predict(test_data)
#    data,label=concatenate_data()
#    accuracy4.append(hist.history['val_acc'] [-1])
#model.save('classification4.h5')



'''
end
--------------------------------------------------------------------------------
'''


#--------------------------------------------------------------------------------
'''
two-way classification
'''
accuracy2=[]
from keras.models import Model
from keras.layers import Dense,Flatten,Input
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras import layers
from keras import optimizers
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from keras import models
from keras.utils import to_categorical
nsr=np.asarray(list(range(0,50)))*10
for ratio in nsr:
    print (f'this is the {ratio}% noise')
    data = np.expand_dims(data, axis=2)
    for i in range(len(label)):
        if label[i]==0:
            label[i]=0
        if label[i]==1:
            label[i]=0
        if label[i]==2:
            label[i]=0
        if label[i]==3:
            label[i]=0
        if label[i]==4:
            label[i]=0
        if label[i]==5:
            label[i]=0
        if label[i]==6:
            label[i]=0
        if label[i]==7:
            label[i]=0  
        if label[i]==8:
            label[i]=0
        if label[i]==9:
            label[i]=1
    label=to_categorical(label)
    for index in range(data.shape[0]):
        data[index,:,:]=(data[index,:].reshape(1,-1)+wgn(data[index,:], ratio/100)*np.random.randn(1,data.shape[1])).T
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1,random_state=40)
    model = Sequential()
    model.add(Conv1D(64, kernel_size=50,strides=50, activation='relu', input_shape=(6000, 1, )))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64,kernel_size=2,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(2, activation='softmax'))
    optimizer = optimizers.adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    model.summary()
    hist=model.fit(train_data, train_label, epochs=50,verbose=1,shuffle=True,validation_data=(test_data,test_label))
    predict=model.predict(test_data)
    data,label=concatenate_data()
    accuracy2.append(hist.history['val_acc'] [-1])



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
plt.ylim(30,105)
plt.xlim(0,500)
plt.tick_params(direction='in',width=1.5,length=4)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.grid(linestyle='--')
#plt.axhline(y=90,c='k',linestyle='--',linewidth=2)
plt.text(105, 72, r'Noise ratio = 100%', fontdict=font1)    

plt.vlines(100,0,105,linestyle='--',linewidth=2)
plt.fill_between(nsr[:np.where(np.asarray(accuracy10)<0.98)[0][0]], 105, color='yellow', alpha=0.2,label='Accuracy>98%')
plt.fill_between(nsr[np.where(np.asarray(accuracy10)<0.98)[0][0]:], 105, color='blue', alpha=0.03,label='Accuracy<98%')

plt.scatter(nsr[:50],np.asarray(accuracy2[:50])*100,color='', marker='o', edgecolors='green')
plt.plot(nsr[:50],np.asarray(accuracy2[:50])*100,linewidth=3,c='green',label='Two-way classification')

plt.plot(nsr[:50],np.asarray(accuracy4[:50])*100,linewidth=3,c='blueviolet',label='Four-way classification')
plt.scatter(nsr[:50],np.asarray(accuracy4[:50])*100,color='', marker='o', edgecolors='blueviolet')

plt.plot(nsr[:50],np.asarray(accuracy10[:50])*100,linewidth=3,c='red',label='Ten-way classification')
plt.scatter(nsr[:50],np.asarray(accuracy10[:50])*100,color='', marker='o', edgecolors='r')
#frame.set_facecolor('white') 
plt.legend(loc='lower center',prop=font2,facecolor ='white')
plt.savefig('noise acc2.jpg',dpi=1200)







#for i in range(len(label)):
#    if label[i]==0:
#        label[i]=0
#    if label[i]==1:
#        label[i]=0
#    if label[i]==2:
#        label[i]=0
#    if label[i]==3:
#        label[i]=0
#    if label[i]==4:
#        label[i]=0
#    if label[i]==5:
#        label[i]=0
#    if label[i]==6:
#        label[i]=0
#    if label[i]==7:
#        label[i]=0  
#    if label[i]==8:
#        label[i]=0
#    if label[i]==9:
#        label[i]=1
#from keras.utils import to_categorical
#label=to_categorical(label)
#from keras.models import Model
#from keras.layers import Dense,Flatten,Input
#from keras.layers import Conv1D
#from keras.layers.pooling import MaxPool1D
#from keras.optimizers import SGD
#from keras import layers
#from keras import optimizers
#from keras.models import Sequential
#from sklearn.cross_validation import train_test_split
#from keras import models
#
#data = np.expand_dims(data, axis=2)
#train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1,random_state=40)
#input=Input(shape=(6000,1))
#conv1=Conv1D(64,kernel_size=50,strides=50,activation='relu')(input)
#pool1=MaxPool1D(pool_size=2)(conv1)
#conv2=Conv1D(64,kernel_size=2,activation='relu')(pool1)
#pool2=MaxPool1D(pool_size=2)(conv2)
#flat1=Flatten(name='flatten')(pool2)
#output=Dense(2,activation='softmax',name='result')(flat1)
#model=Model(inputs=input,outputs=output)
#model.summary()
#optimizer = optimizers.adam(lr=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
#model.summary()
#model.fit(train_data, train_label, epochs=50,verbose=1,shuffle=True,validation_data=(test_data,test_label))
#
#predict=model.predict(test_data)
#model.save('classification2.h5')
'''
end
'''
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
'''
confusion matrix
'''
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
#predict=predict.argmax(1)               #one-hot 
#cm=confusion_matrix(test_label,predict) #confusion matrix
#print (cm)
#
#cm_df = pd.DataFrame(cm,
#                     index = ['','','','','','','','','',''],
#                     columns = ['','','','','','','','','',''])
#plt.gca().yaxis.set_tick_params(rotation=0)
#sns.heatmap(cm_df, annot=True, fmt='d',cmap='Blues')
#plt.gca().yaxis.set_tick_params(rotation=90)
#
#plt.gca().xaxis.tick_top()
#plt.xlabel('False Label')
#plt.ylabel('True Label')
#
#plt.savefig('cwru10.jpg',dpi=1200)


