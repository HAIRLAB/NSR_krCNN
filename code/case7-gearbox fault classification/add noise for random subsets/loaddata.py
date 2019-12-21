# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:11:04 2018

@author: 马小跳
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
    
data1_vib = readfile("F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\断一个齿1\振动\恒转速")
data1_cur = readfile("F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\断一个齿1\电流\恒转速")

data2_vib=readfile("F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\断一个齿2\振动数据10.31")
data2_cur = readfile("F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\断一个齿2\电流数据10.31")
data2_vib=data2_vib[5:11]+data2_vib[13:]
data2_cur=data2_cur[2:5]+data2_cur[6:]

data3_vib=readfile(path="F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\双轴断齿\第一组数据 定转速+变转速\振动")
data3_vib=data3_vib[:3]+data3_vib[4:9]+data3_vib[10:]
data3_cur=readfile("F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\双轴断齿\第一组数据 定转速+变转速\电流")
data3_cur=data3_cur[:3]+data3_cur[4:9]+data3_cur[10:]

data4_vib=readfile("F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\正常齿轮\振动")
data4_cur=readfile("F:\江南大学齿轮箱\齿轮箱数据\齿轮箱数据\正常齿轮\电流")

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
   







