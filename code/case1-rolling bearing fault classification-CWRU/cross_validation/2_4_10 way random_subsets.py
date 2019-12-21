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
#data,label=concatenate_data()


'''
ten-way classification
'''
#from keras.utils import to_categorical
#label=to_categorical(label)
#from keras.models import Model
#from keras.layers import Dense,Flatten,Input
#from keras.layers import Conv1D, MaxPooling1D
#from keras.optimizers import SGD
#from keras import layers
#from keras import optimizers
#from keras.models import Sequential
#from sklearn.cross_validation import train_test_split
#from keras import models
#
#data = np.expand_dims(data, axis=2)
#train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1,random_state=40)
#model = Sequential()
#model.add(Conv1D(64, kernel_size=50,strides=50, activation='relu', input_shape=(6000, 1, )))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(64,kernel_size=2,activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten(name='flatten'))
#model.add(Dense(10, activation='softmax'))
#optimizer = optimizers.adam(lr=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
#model.summary()
#model.fit(train_data, train_label, epochs=50,verbose=1,shuffle=True,validation_data=(test_data,test_label))
#predict=model.predict(test_data)
#model.save('classification10.h5')
'''
end
'''
#--------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
'''
four-way classification
'''
#for i in range(len(label)):
#    if label[i]==0:
#        label[i]=0
#    if label[i]==1:
#        label[i]=0
#    if label[i]==2:
#        label[i]=0
#    if label[i]==3:
#        label[i]=1
#    if label[i]==4:
#        label[i]=1
#    if label[i]==5:
#        label[i]=1
#    if label[i]==6:
#        label[i]=2
#    if label[i]==7:
#        label[i]=2  
#    if label[i]==8:
#        label[i]=2
#    if label[i]==9:
#        label[i]=3
#
#from keras.utils import to_categorical
#label=to_categorical(label)
#from keras.models import Model
#from keras.layers import Dense,Flatten,Input
#from keras.layers import Conv1D, MaxPooling1D
#from keras.optimizers import SGD
#from keras import layers
#from keras import optimizers
#from keras.models import Sequential
#from sklearn.cross_validation import train_test_split
#from keras import models
#
#data = np.expand_dims(data, axis=2)
#train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1,random_state=40)
#model = Sequential()
#model.add(Conv1D(64, kernel_size=50,strides=50, activation='relu', input_shape=(6000, 1, )))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(64,kernel_size=2,activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten(name='flatten'))
#model.add(Dense(4, activation='softmax'))
#optimizer = optimizers.adam(lr=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
#model.summary()
#model.fit(train_data, train_label, epochs=50,verbose=1,shuffle=True,validation_data=(test_data,test_label))
#predict=model.predict(test_data)
#model.save('classification4.h5')

'''
end
--------------------------------------------------------------------------------
'''


#--------------------------------------------------------------------------------
'''
two-way classification
'''
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

plt.plot(data[11,:],c='k')
plt.savefig('raw.jpg',dpi=200)



#plt.plot(range(6000),data[10,:6000],c='r')
#plt.plot(range(6000,12000),data[10,6000:12000],c='k')
#plt.plot(range(12000,18000),data[10,12000:18000],c='g')






