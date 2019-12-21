# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:11:04 2018

"""
import pandas as pd
import numpy as np
import os
import scipy.io as sio

def readfile(path):
    data=[]
    files=os.listdir(path)
    for txt in files:
        temp=pd.read_table(open(os.path.join(path,txt)))
        temp=np.asarray(temp)
        temp=temp[:,1:]  
        data.append(temp)
    return data
    
data1_vib = readfile("vib")
data1_cur = readfile("current")

data2_vib=readfile("vib2")
data2_cur = readfile("current2")
data2_vib=data2_vib[5:11]+data2_vib[13:]
data2_cur=data2_cur[2:5]+data2_cur[6:]

data3_vib=readfile(path="vib3")
data3_vib=data3_vib[:3]+data3_vib[4:9]+data3_vib[10:]
data3_cur=readfile("current3")
data3_cur=data3_cur[:3]+data3_cur[4:9]+data3_cur[10:]

data4_vib=readfile("vib_normal")
data4_cur=readfile("current_normal")

def segmentation1(data):
    temp=[]
    for item in data:
        a=np.reshape(item[:,1][:item[:,1].shape[0]//6000*6000],(item[:,1].shape[0]//6000,6000))
        temp.append(a)
    output=temp[0]
    for item in temp[1:]:
        output=np.concatenate((output,item),axis=0)
    return output
def segmentation2(data):
    temp=[]
    for item in data:
        a=np.reshape(item[:,0][:item[:,0].shape[0]//6000*6000],(item[:,0].shape[0]//6000,6000))
        temp.append(a)
    output=temp[0]
    for item in temp[1:]:
        output=np.concatenate((output,item),axis=0)
    return output


data1_vib=segmentation1(data1_vib)
data2_vib=segmentation2(data2_vib)
data3_vib=segmentation1(data3_vib)
data4_vib=segmentation1(data4_vib)

data1_cur=segmentation1(data1_cur)
data2_cur=segmentation2(data2_cur)
data3_cur=segmentation2(data3_cur)
data4_cur=segmentation2(data4_cur)    
   







