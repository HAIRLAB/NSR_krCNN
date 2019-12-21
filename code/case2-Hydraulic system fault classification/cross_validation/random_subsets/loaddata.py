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
profile = np.loadtxt('profile.txt')
ps1 = np.loadtxt('PS1.txt')
ps2 = np.loadtxt('PS2.txt')
ps3 = np.loadtxt('PS3.txt')
ps4 = np.loadtxt('PS4.txt')
ps5 = np.loadtxt('PS5.txt')
ps6 = np.loadtxt('PS6.txt')
eps1 = np.loadtxt('EPS1.txt')
fs1 = np.loadtxt('FS1.txt')
fs2 = np.loadtxt('FS2.txt')
ts1 = np.loadtxt('TS1.txt')
ts2 = np.loadtxt('TS2.txt')
ts3 = np.loadtxt('TS3.txt')
ts4 = np.loadtxt('TS4.txt')
vs1 = np.loadtxt('VS1.txt')
ce = np.loadtxt('CE.txt') 
cp = np.loadtxt('CP.txt') 
se = np.loadtxt('SE.txt')

for i in range(20):
    plt.plot(vs1[i,:],label=i)
plt.legend()


label0=profile[:,0]
for i in range(len(label0)):
    if label0[i]==3:
        label0[i]=0
    if label0[i]==20:
        label0[i]=1    
    if label0[i]==100:
        label0[i]=2

label1=profile[:,1]
for i in range(len(label1)):
    if label1[i]==100:
        label1[i]=0
    if label1[i]==90:
        label1[i]=1    
    if label1[i]==80:
        label1[i]=2
    if label1[i]==73:
        label1[i]=3

label2=profile[:,2]
for i in range(len(label2)):
    if label2[i]==0:
        label2[i]=0
    if label2[i]==1:
        label2[i]=1    
    if label2[i]==2:
        label2[i]=2

label3=profile[:,3]
for i in range(len(label3)):
    if label3[i]==130:
        label3[i]=0
    if label3[i]==115:
        label3[i]=1    
    if label3[i]==100:
        label3[i]=2
    if label3[i]==90:
        label3[i]=3

label4=profile[:,4]
for i in range(len(label4)):
    if label4[i]==0:
        label4[i]=0
    if label4[i]==1:
        label4[i]=1    

from keras.utils import to_categorical
label0=to_categorical(label0)
label1=to_categorical(label1)
label2=to_categorical(label2)
label3=to_categorical(label3)
label4=to_categorical(label4)





