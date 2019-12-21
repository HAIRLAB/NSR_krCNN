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
    train_temp=[]
    test_temp=[]
    train_label=[]
    test_label=[]
    for mat in files:
        if '28' not in mat:
            if '_1' in mat:
                temp1=sio.loadmat(os.path.join(path,mat))
                for key in temp1.keys():
                    if 'DE' in key:
                        test_temp.append(temp1[key][:120000])
                if 'B' in mat:
                    if '07' in mat:                           
                        test_label.append([0]*20)
                    if '14' in mat:
                        test_label.append([1]*20)
                    if '21' in mat:
                        test_label.append([2]*20)
                if 'IR' in mat:
                    if '07' in mat:                           
                        test_label.append([3]*20)
                    if '14' in mat:
                        test_label.append([4]*20)
                    if '21' in mat:
                        test_label.append([5]*20)
                if 'OR' in mat:
                    if '07' in mat:                           
                        test_label.append([6]*20)
                    if '14' in mat:
                        test_label.append([7]*20)
                    if '21' in mat:
                        test_label.append([8]*20)
                        
            elif '_1' not in mat:
                temp1=sio.loadmat(os.path.join(path,mat))
                for key in temp1.keys():
                    if 'DE' in key:
                        train_temp.append(temp1[key][:120000])
                if 'B' in mat:
                    if '07' in mat:                           
                        train_label.append([0]*20)
                    if '14' in mat:
                        train_label.append([1]*20)
                    if '21' in mat:
                        train_label.append([2]*20)
                if 'IR' in mat:
                    if '07' in mat:                           
                        train_label.append([3]*20)
                    if '14' in mat:
                        train_label.append([4]*20)
                    if '21' in mat:
                        train_label.append([5]*20)
                if 'OR' in mat:
                    if '07' in mat:                           
                        train_label.append([6]*20)
                    if '14' in mat:
                        train_label.append([7]*20)
                    if '21' in mat:
                        train_label.append([8]*20)                
                                                                                  
    train_temp=np.asarray(train_temp)
    train_data=train_temp.reshape((-1,6000))
    train_label=np.asarray(train_label)
    train_label=train_label.reshape((-1,1))
    
    test_temp=np.asarray(test_temp)
    test_data=test_temp.reshape((-1,6000))
    test_label=np.asarray(test_label)
    test_label=test_label.reshape((-1,1))    
    
    return train_data, test_data, train_label, test_label

def load_normal_data():
    '''
    Load normal_data of bearing
    we use DE variable for working
    Reshape the raw data in a format of (samples,6000)
    '''    
    
    path='Normal_Baseline_Data'
    files=os.listdir(path)
    train_temp2=[]
    test_temp2=[]
    train_label2=[]
    test_label2=[]
    for mat in files:
        if '_1' in mat:      
            temp1=sio.loadmat(os.path.join(path,mat))
            for key in temp1.keys():
                if 'DE' in key:
                    if 240000<len(temp1[key])<480000:
                        test_temp2.append(temp1[key][:240000])
                    if len(temp1[key])>480000:
                        test_temp2.append(temp1[key][:480000])
        elif '_1' not in mat:      
            temp1=sio.loadmat(os.path.join(path,mat))
            for key in temp1.keys():
                if 'DE' in key:
                    if 240000<len(temp1[key])<480000:
                        train_temp2.append(temp1[key][:240000])
                    if len(temp1[key])>480000:
                        train_temp2.append(temp1[key][:480000])
                        
    train_data2=np.concatenate((train_temp2[0],train_temp2[1],train_temp2[2]))
    train_data2=train_data2.reshape((-1,6000))
    test_data2=test_temp2[0]
    test_data2=test_data2.reshape((-1,6000))   
    test_label2=np.ones((test_data2.shape[0],1))*9
    train_label2=np.ones((train_data2.shape[0],1))*9
    
    return train_data2,test_data2,train_label2,test_label2




def concatenate_data():
    '''
    combine all data to be a set
    '''    
    train_data1, test_data1, train_label1, test_label1=load_fault_data()
    train_data2,test_data2,train_label2,test_label2=load_normal_data()    
    train_data=np.concatenate((train_data1,train_data2))
    train_label=np.concatenate((train_label1,train_label2))
    test_data=np.concatenate((test_data1,test_data2))
    test_label=np.concatenate((test_label1,test_label2))
    return train_data,train_label,test_data,test_label
train_data,train_label,test_data,test_label=concatenate_data()


'''
ten-way classification
'''
from keras.utils import to_categorical
train_label=to_categorical(train_label)
test_label=to_categorical(test_label)

from keras.models import Model
from keras.layers import Dense,Flatten,Input
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras import layers
from keras import optimizers
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from keras import models
#from sklearn.preprocessing import StandardScaler  
#minmaxTransformer = StandardScaler()
#train_data = minmaxTransformer.fit_transform(train_data)
#test_data=minmaxTransformer.transform(test_data)
#
train_data=np.expand_dims(train_data,axis=2)
test_data=np.expand_dims(test_data,axis=2)

model = Sequential()
model.add(Conv1D(64, kernel_size=50,strides=50, activation='tanh', input_shape=(6000, 1, )))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64,kernel_size=2,activation='tanh'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten(name='flatten'))
model.add(Dense(10, activation='softmax'))
optimizer = optimizers.adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
model.summary()
model.fit(train_data, train_label, epochs=50,verbose=1,shuffle=True,validation_data=(test_data,test_label))
predict=model.predict(test_data)
'''
end
'''
#--------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
'''
four-way classification
'''
#for i in range(len(train_label)):
#    if train_label[i]==0:
#        train_label[i]=0
#    if train_label[i]==1:
#        train_label[i]=0
#    if train_label[i]==2:
#        train_label[i]=0
#    if train_label[i]==3:
#        train_label[i]=1
#    if train_label[i]==4:
#        train_label[i]=1
#    if train_label[i]==5:
#        train_label[i]=1
#    if train_label[i]==6:
#        train_label[i]=2
#    if train_label[i]==7:
#        train_label[i]=2  
#    if train_label[i]==8:
#        train_label[i]=2
#    if train_label[i]==9:
#        train_label[i]=3
#        
#for i in range(len(test_label)):
#    if test_label[i]==0:
#        test_label[i]=0
#    if test_label[i]==1:
#        test_label[i]=0
#    if test_label[i]==2:
#        test_label[i]=0
#    if test_label[i]==3:
#        test_label[i]=1
#    if test_label[i]==4:
#        test_label[i]=1
#    if test_label[i]==5:
#        test_label[i]=1
#    if test_label[i]==6:
#        test_label[i]=2
#    if test_label[i]==7:
#        test_label[i]=2  
#    if test_label[i]==8:
#        test_label[i]=2
#    if test_label[i]==9:
#        test_label[i]=3
#
#from keras.utils import to_categorical
#train_label=to_categorical(train_label)
#test_label=to_categorical(test_label)
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
#train_data=np.expand_dims(train_data,axis=2)
#test_data=np.expand_dims(test_data,axis=2)
#model = Sequential()
#model.add(Conv1D(64, kernel_size=50,strides=50, activation='tanh', input_shape=(6000, 1, )))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(64,kernel_size=2,activation='tanh'))
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
#for i in range(len(train_label)):
#    if train_label[i]!=9:
#        train_label[i]=0
#    if train_label[i]==9:
#        train_label[i]=1
#for i in range(len(test_label)):
#    if test_label[i]!=9:
#        test_label[i]=0
#    if test_label[i]==9:
#        test_label[i]=1        
#        
#from keras.utils import to_categorical
#train_label=to_categorical(train_label)
#test_label=to_categorical(test_label)
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
#train_data=np.expand_dims(train_data,axis=2)
#test_data=np.expand_dims(test_data,axis=2)
#
#input=Input(shape=(6000,1))
#conv1=Conv1D(64,kernel_size=50,strides=50,activation='tanh')(input)
#pool1=MaxPool1D(pool_size=2)(conv1)
#conv2=Conv1D(64,kernel_size=2,activation='tanh')(pool1)
#pool2=MaxPool1D(pool_size=2)(conv2)
#flat1=Flatten(name='flatten')(pool2)
#output=Dense(2,activation='softmax',name='result')(flat1)
#model=Model(inputs=input,outputs=output)
#model.summary()
#optimizer = optimizers.adam(lr=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
#model.summary()
#model.fit(train_data, train_label, epochs=50,verbose=1,shuffle=True,validation_data=(test_data,test_label))
#predict=model.predict(test_data)
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

#plt.plot(data[11,:],c='k')
#plt.savefig('raw.jpg',dpi=200)



#plt.plot(range(6000),data[10,:6000],c='r')
#plt.plot(range(6000,12000),data[10,6000:12000],c='k')
#plt.plot(range(12000,18000),data[10,12000:18000],c='g')






