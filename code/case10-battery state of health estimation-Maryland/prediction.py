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
The data is from the results of loaddata.py
"""
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(10)

data=np.concatenate((voltage_36,voltage_38,voltage_37,voltage_35))  # change the position when use a CV

label_conc=np.concatenate((capacity_36,capacity_38,capacity_37,capacity_35)) # change the position when use a CV
label=label_conc/(np.max(label_conc)+0.0000001)

predict_len=932  # settting

from sklearn import preprocessing
scaler= preprocessing.StandardScaler().fit(data[:-predict_len,:])
train_data=scaler.transform(data[:-predict_len,:])
test_data=scaler.transform(data[-predict_len:,:])

train_data=np.expand_dims(train_data,axis=2)
test_data=np.expand_dims(test_data,axis=2)

train_label=label[:-predict_len]
test_label=label[-predict_len:]


from keras.models import load_model
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers

input=Input(shape=(404,1))
conv1=Conv1D(64,kernel_size=10,strides=10,activation='relu')(input)
pool1=MaxPool1D(pool_size=2)(conv1)
conv2=Conv1D(64,kernel_size=2,activation='relu')(pool1)
pool2=MaxPool1D(pool_size=2)(conv2)
flat1=Flatten(name='flatten')(pool2)
dense2=Dense(500,activation='relu')(flat1)
dense3=Dense(50,activation='relu')(dense2)
output=Dense(1,activation='tanh',name='result')(dense3)
model=Model(inputs=input,outputs=output)
model.summary()
optimizer = optimizers.adam(lr=0.0001,decay=1e-6)
model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=[])
model.summary()
history=model.fit(train_data,train_label, epochs=100,validation_data=(test_data,test_label))

prediction=model.predict(test_data)

#from keras.models import load_model
#model=load_model('38.h5')
#model.save('38.h5')

plt.scatter(range(predict_len),label[-predict_len:]*np.max(label_conc),c='r')
plt.scatter(range(predict_len),prediction[-predict_len:]*np.max(label_conc),c='g')
plt.xlabel('Cycle number')
plt.ylabel('Capacity')

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

mse=mean_squared_error(test_label,prediction)
mae=mean_absolute_error(test_label,prediction)
r2=r2_score(test_label,prediction)
rmse= np.sqrt(mse)
print ('maryland 35')
print ('mse:',mse)
print ('mae:',mae)
print ('r2:',r2)
print ('rmse:',rmse)


