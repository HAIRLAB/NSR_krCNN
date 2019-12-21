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
three-way, four-way, three-way and four-way classification
use the ps1 data results of loaddata.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
sss=StratifiedShuffleSplit(test_size=0.1, random_state=40)
"""
Cooler condition classification, three-way
"""
#for train_index,test_index in sss.split(ps1,label0):
#     train_ps1=ps1[train_index]
#     train_label=label0[train_index]
#     test_ps1=ps1[test_index]
#     test_label=label0[test_index]
"""
Valve condition classification, four-way
"""
#for train_index,test_index in sss.split(ps1,label1):
#     train_ps1=ps1[train_index]
#     train_label=label1[train_index]
#     test_ps1=ps1[test_index]
#     test_label=label1[test_index]
"""
Internal pump leakage classification, three-way 
"""
#for train_index,test_index in sss.split(ps1,label2):
#     train_ps1=ps1[train_index]
#     train_label=label2[train_index]
#     test_ps1=ps1[test_index]
#     test_label=label2[test_index]
"""
Hydraulic accumulator classification, four-way
"""
for train_index,test_index in sss.split(ps1,label3):
    train_ps1=ps1[train_index]
    train_label=label3[train_index]
    test_ps1=ps1[test_index]
    test_label=label3[test_index]

from sklearn import preprocessing
scaler= preprocessing.StandardScaler().fit(train_ps1)
train_ps1_norm=scaler.transform(train_ps1)
test_ps1_norm=scaler.transform(test_ps1)
train_ps1_norm=np.expand_dims(train_ps1_norm,axis=2)
test_ps1_norm=np.expand_dims(test_ps1_norm,axis=2)
print (train_ps1_norm.shape)

from keras.models import load_model
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

input=Input(shape=(6000,1))
conv1=Conv1D(64,kernel_size=10,strides=5,activation='relu')(input)
pool1=MaxPool1D(pool_size=2)(conv1)
conv2=Conv1D(64,kernel_size=2,activation='relu')(pool1)
pool2=MaxPool1D(pool_size=2)(conv2)
flat1=Flatten(name='flatten')(pool2)
dense2=Dense(500,activation='relu')(flat1)
dense3=Dense(50,activation='relu')(dense2)
output=Dense(4,activation='softmax',name='result')(dense3)  # change the node in different classificaiton ways
model=Model(inputs=input,outputs=output)
model.summary()
optimizer = optimizers.adam(lr=0.00001,decay=1e-5)
model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc'])
model.summary()
history=model.fit(train_ps1_norm,train_label, epochs=1000,validation_data=(test_ps1_norm,test_label),verbose=2)
predict= model.predict(test_ps1_norm)

for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        if predict[i,j]<0.5:
            predict[i, j]=0
        elif predict[i,j]>=0.5:
            predict[i,j]=1
test_label=test_label.argmax(1)
predict=predict.argmax(1)               #one-hot
cm=confusion_matrix(test_label,predict) #confusion matrix
print(cm)
cm_df = pd.DataFrame(cm,
                  index = ['setosa','versicolor','virginica'],
                  columns = ['setosa','versicolor','virginica'])
cmap = plt.get_cmap('Blues')

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
plt.savefig('bbb.jpg',dpi=1200)
plt.show()

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('bbb.jpg',dpi=1200)
plt.show()


