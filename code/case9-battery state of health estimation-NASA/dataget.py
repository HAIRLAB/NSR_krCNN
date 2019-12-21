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
used for extracting data from NASA batteries
"""
import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt

def dataget(path):
    '''
    extract charge and discharge signals.
    charge_voltage_charge is the voltage signal of charging 
    discharge_voltage_charge is the voltage signal of discharging 
     path :
         the path of four different batteries
    
    '''
    data=sio.loadmat(path)
    data=data[list(data.keys())[-1]]
    len_data=data[0][0][0].shape[1]
    capacity=[]
    charge_Voltage_measured=[]
    charge_Current_measured=[]
    charge_Temperature_measured=[]
    charge_Current_charge=[]
    charge_Voltage_charge=[]
    charge_Time=[]
    discharge_Voltage_measured=[]
    discharge_Current_measured=[]
    discharge_Temperature_measured=[]
    discharge_Current_load=[]
    discharge_Voltage_load=[]
    dicharge_Time=[]

    for i in range(len_data):
        if 'charge' == data[0][0][0][0][i][0][0]:
            charge_Voltage_measured.append(data[0][0][0][0][i][3][0][0][0][0,:])
            charge_Current_measured.append(data[0][0][0][0][i][3][0][0][1][0,:])
            charge_Temperature_measured.append(data[0][0][0][0][i][3][0][0][2][0,:])
            charge_Current_charge.append(data[0][0][0][0][i][3][0][0][3][0,:])
            charge_Voltage_charge.append(data[0][0][0][0][i][3][0][0][4][0,:])
            charge_Time.append(data[0][0][0][0][i][3][0][0][5][0,:])
        if 'discharge' == data[0][0][0][0][i][0][0]:
            capacity.append(data[0][0][0][0][i][3][0][0][-1][0,:])
            discharge_Voltage_measured.append(data[0][0][0][0][i][3][0][0][0][0,:])
            discharge_Current_measured.append(data[0][0][0][0][i][3][0][0][1][0,:])
            discharge_Temperature_measured.append(data[0][0][0][0][i][3][0][0][2][0,:])
            discharge_Current_load.append(data[0][0][0][0][i][3][0][0][3][0,:])
            discharge_Voltage_load.append(data[0][0][0][0][i][3][0][0][4][0,:])
            dicharge_Time.append(data[0][0][0][0][i][3][0][0][5][0,:])
    capacity=np.asarray(capacity).reshape(-1)
    del charge_Voltage_measured[12]
    del charge_Voltage_measured[-1]
###pre-processing for charge_voltage_measured 
######## normalize the length of charge_Voltage_measured
    charlen=[] 
    charge_pre=[]       
    for item in charge_Voltage_measured:
        charlen.append(len(item))
    maxlen=np.max(charlen)
    for item in charge_Voltage_measured:
        charge_pre.append(np.concatenate((item,4.202*np.ones((maxlen-len(item),1))[:,0])))            
    charge_Voltage_measured=charge_pre
      
###pre-processing for discharge_voltage_measured    
###remove the rising part of discharge_voltage_measured 
    temp=[]
    for item in discharge_Voltage_measured:
        for i in range(100,len(item)):
            if item[i-1]>=item[i]:
                if i==len(item)-1:
                    temp.append(item[:i+1])
            if item[i-1]<item[i]:
                temp.append(item[:i])
                break
##############to find the max shape of discharge_voltage_measured            
    prelen=[]
    for item in temp:
        prelen.append(len(item))   
    pre_maxlen=np.max(prelen)
######## nomalize the length of discharge_voltage_measured
    discharge_Voltage_measured=[]
    for item in temp:
        discharge_Voltage_measured.append(np.concatenate((4.199*np.ones((pre_maxlen-len(item),1))[:,0],item)))   
    return capacity,charge_Voltage_measured,discharge_Voltage_measured

B06_cap,B06_char_V_mea,B06_dischar_V_mea=dataget(path='B0006.mat')
B05_cap,B05_char_V_mea,B05_dischar_V_mea=dataget(path='B0005.mat')
B07_cap,B07_char_V_mea,B07_dischar_V_mea=dataget(path='B0007.mat')
B18_cap,B18_char_V_mea,B18_dischar_V_mea=dataget(path='B0018.mat')

#######delete the abnormal data
B18_cap=np.delete(B18_cap,[7],axis=0)
del B18_char_V_mea[7]
del B18_dischar_V_mea[7]
####################################

def unify(data1,data2,data3,data4):
####to unify the shape of B06, B05 and B07
    maxlen=np.max([len(data1[0]),len(data2[0]),len(data3[0]),len(data4[0])])
    temp=[]
    for item in data1:
        temp.append(np.concatenate((4.199*np.ones((maxlen-len(item),1))[:,0],item)))
    data1=temp
    temp=[]
    for item in data2:
        temp.append(np.concatenate((4.199*np.ones((maxlen-len(item),1))[:,0],item)))
    data2=temp    
    temp=[]
    for item in data3:
        temp.append(np.concatenate((4.199*np.ones((maxlen-len(item),1))[:,0],item)))
    data3=temp 
    temp=[]
    for item in data4:
        temp.append(np.concatenate((4.199*np.ones((maxlen-len(item),1))[:,0],item)))
    data4=temp     
    return data1,data2,data3,data4
B05_dischar_V_mea,B06_dischar_V_mea,B07_dischar_V_mea,B18_dischar_V_mea=unify(B05_dischar_V_mea,
                                                                              B06_dischar_V_mea,B07_dischar_V_mea,B18_dischar_V_mea)

B05_dischar_V_mea=np.asarray(B05_dischar_V_mea)
B06_dischar_V_mea=np.asarray(B06_dischar_V_mea)
B07_dischar_V_mea=np.asarray(B07_dischar_V_mea)
B18_dischar_V_mea=np.asarray(B18_dischar_V_mea)

plt.plot(B05_cap)
plt.plot(B06_cap)
plt.plot(B07_cap)








