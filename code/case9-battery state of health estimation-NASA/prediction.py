'''
------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by GuiJ Ma on 2018-08-30
#
# ------------------------------------------------------------------------
'''

"""
The variable is from dataget.py
use a discharge voltage as input 
use discharge capacity as output
"""

import matplotlib.pyplot as plt
import numpy as np
import random 
random.seed(10)

data=np.concatenate((B05_dischar_V_mea,B07_dischar_V_mea,B06_dischar_V_mea)) #need to change the position when use a CV
label_conc=np.concatenate((B05_cap,B07_cap,B06_cap))  #need to change the position when use a CV
label=label_conc/(np.max(label_conc)+0.0000001)


data_length=168
train_data=data[:-data_length,:]
test_data=data[-data_length:,:]
train_label=label[:-data_length]
test_label=label[-data_length:]

from sklearn import preprocessing
scaler= preprocessing.StandardScaler().fit(train_data)
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)
train_data=np.expand_dims(train_data,axis=2)
test_data=np.expand_dims(test_data,axis=2)



from keras.models import load_model
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers

input=Input(shape=(371,1))
conv1=Conv1D(64,kernel_size=5,strides=5,activation='relu')(input)
pool1=MaxPool1D(pool_size=2)(conv1)
conv2=Conv1D(64,kernel_size=2,activation='relu')(pool1)
pool2=MaxPool1D(pool_size=2)(conv2)
conv3=Conv1D(64,kernel_size=2,activation='relu')(pool2)
pool3=MaxPool1D(pool_size=2)(conv3)

flat1=Flatten(name='flatten')(pool2)

dense2=Dense(500,activation='relu')(flat1)
dense3=Dense(50,activation='relu')(dense2)

output=Dense(1,activation='sigmoid',name='result')(dense3)
model=Model(inputs=input,outputs=output)
model.summary()

optimizer = optimizers.adam(lr=0.0001,decay=1e-6)
model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=[])
model.summary()
history=model.fit(train_data,train_label, epochs=1000,validation_data=(test_data,test_label))

prediction=model.predict(test_data)

#from keras.models import load_model
#model=load_model('56-7.h5')
#model.save('67-5.h5')

plt.plot(test_label*np.max(label_conc))
plt.plot(prediction*np.max(label_conc))
plt.xlabel('Cycle number')
plt.ylabel('Capacity')


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

mse=mean_squared_error(test_label,prediction)
mae=mean_absolute_error(test_label,prediction)
r2=r2_score(test_label,prediction)
rmse= np.sqrt(mse)
print ('NASA 57-6')
print ('mse:',mse)
print ('mae:',mae)
print ('r2:',r2)
print ('rmse:',rmse)


predict6=prediction*np.max(label_conc)
label6=test_label*np.max(label_conc)

plt.plot(predict6)
plt.plot(label6)
plt.plot(predict5)
plt.plot(label5)
plt.plot(predict7)
plt.plot(label7)





