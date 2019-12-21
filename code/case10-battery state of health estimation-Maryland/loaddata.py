'''
------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by GuiJ Ma on 2018-08-30
#
# ------------------------------------------------------------------------
'''

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
def loaddata35(path):
    """
    load data Battery 35
    """
    files=os.listdir(path)
    dir_list = sorted(files,  key=lambda x: os.path.getmtime(os.path.join(path, x)))  ##sorted by modified time
    
        capacity1=[]
        voltage2=[]
        cycle_index=[]
        voltage=[]
        capacity=[]
        voltage1=[]
        path_xlsx=os.path.join(path,dir_list[2])
        df = pd.read_excel(path_xlsx, None)
        a=df['Channel_1-008']
        b=np.asarray(a)
        for i in range(b.shape[0]):
            cycle_index.append(b[i][5])
            voltage.append(b[i][7])
            capacity.append(b[i][9])
            
        for index in np.unique(cycle_index):
            start_index=0
            for temp in range(index):
                start_index=cycle_index.count(temp)+start_index
            capacity1.append(capacity[start_index+cycle_index.count(index)-1]-capacity[start_index])
        ## delete the first charge curve       
            for vol_index in range(start_index+15,start_index+cycle_index.count(index)):
                if voltage[vol_index]<voltage[vol_index-1]:  
                    voltage1.append(voltage[vol_index:start_index+cycle_index.count(index)])
                    break
        ## delete the latter charge curve        
        for item in voltage1:
            for i in range(30,len(item)):
                if i !=len(item)-1:
                    if item[i]>=item[i-1]:
                        voltage2.append(item[:i])
                        break
                else :
                    if item[i]<item[i-1]:
                        voltage2.append(item[:i+1])
                        break 
    ###some capacity is empty need to be deleted
        if len(capacity1)!=len(voltage2):
            del capacity1[-1]
    return voltage2,capacity1
voltage_35,capacity_35=loaddata35(path='path35')      

def loaddata36(path):
    """
    load data Battery 36
    """
    files=os.listdir(path)
    dir_list = sorted(files,  key=lambda x: os.path.getmtime(os.path.join(path, x)))  ##sorted by modified time
    
    capacity1=[]
    voltage2=[]
    
    for ls in dir_list:
        cycle_index=[]
        voltage=[]
        capacity=[]
        voltage1=[]
        path_xlsx=os.path.join(path,ls)
        
        df = pd.read_excel(path_xlsx, None)
        a=df['Channel_1-009']
        b=np.asarray(a)
        for i in range(b.shape[0]):
            cycle_index.append(b[i][5])
            voltage.append(b[i][7])
            capacity.append(b[i][9])
            
        for index in np.unique(cycle_index):
            start_index=0
            for temp in range(index):
                start_index=cycle_index.count(temp)+start_index
            capacity1.append(capacity[start_index+cycle_index.count(index)-1]-capacity[start_index])
        ## delete the first charge curve       
            for vol_index in range(start_index+15,start_index+cycle_index.count(index)):
                if voltage[vol_index]<voltage[vol_index-1]:  
                    voltage1.append(voltage[vol_index:start_index+cycle_index.count(index)])
                    break
        ## delete the latter charge curve        
        for item in voltage1:
            for i in range(30,len(item)):
                if i !=len(item)-1:
                    if item[i]>=item[i-1]:
                        voltage2.append(item[:i])
                        break
                else :
                    if item[i]<item[i-1]:
    #                    plt.plot(item[:i])
                        voltage2.append(item[:i+1])
                        break 
    
    ###some capacity is empty need to be deleted
        if ls=='5.xlsx':
            del capacity1[-1]
            del voltage2[-1]
        if ls=='9.xlsx':
            del capacity1[-1]
            del voltage2[-1]    
        if ls=='10.xlsx':
            del capacity1[-1]
        if ls=='14.xlsx':
            del capacity1[-1]
        if ls=='17.xlsx':
            del capacity1[-1]
            del voltage2[-1]
        if ls=='21.xlsx':
            del capacity1[-1]
        if ls=='26.xlsx':
            del capacity1[-17]
            del capacity1[-33] 
    return voltage2,capacity1
voltage_36,capacity_36=loaddata36(path='path36')      


def loaddata37(path):
    """
    load data Battery 37
    """
    files=os.listdir(path)
    dir_list = sorted(files,  key=lambda x: os.path.getmtime(os.path.join(path, x)))  ##sorted by modified time
    
    capacity1=[]
    voltage2=[]
    
    for ls in dir_list:
        cycle_index=[]
        voltage=[]
        capacity=[]
        voltage1=[]
        path_xlsx=os.path.join(path,ls)        
        df = pd.read_excel(path_xlsx, None)
        a=df['Channel_1-010']
        b=np.asarray(a)
        for i in range(b.shape[0]):
            cycle_index.append(b[i][5])
            voltage.append(b[i][7])
            capacity.append(b[i][9])
            
        for index in np.unique(cycle_index):
            start_index=0
            for temp in range(index):
                start_index=cycle_index.count(temp)+start_index
            capacity1.append(capacity[start_index+cycle_index.count(index)-1]-capacity[start_index])
        ## delete the first charge curve       
            for vol_index in range(start_index+15,start_index+cycle_index.count(index)):
                if voltage[vol_index]<voltage[vol_index-1]:  
                    voltage1.append(voltage[vol_index:start_index+cycle_index.count(index)])
                    break
        ## delete the latter charge curve        
        for item in voltage1:
            for i in range(30,len(item)):
                if i !=len(item)-1:
                    if item[i]>=item[i-1]:
                        voltage2.append(item[:i])
                        break
                else :
                    if item[i]<item[i-1]:
                        voltage2.append(item[:i+1])
                        break                                   
    
    ###some capacity is empty need to be deleted
        if ls=='5.xlsx':
            del capacity1[-1]
            del voltage2[-1] 
        if ls=='9.xlsx':
            del capacity1[-1]       
        if ls=='14.xlsx':
            del capacity1[-1] 
        if ls=='18.xlsx':
            del capacity1[-1]  
        if ls=='22.xlsx':
            del capacity1[-1]  
        if ls=='26.xlsx':
            del capacity1[-1]
    return voltage2,capacity1
voltage_37,capacity_37=loaddata37(path='path37')      



def loaddata38(path):
    """
    load data of Battery 38
    """
    files=os.listdir(path)
    dir_list = sorted(files,  key=lambda x: os.path.getmtime(os.path.join(path, x)))  ##sorted by modified time
  
    capacity1=[]
    voltage2=[]
    
    for ls in dir_list:
        cycle_index=[]
        voltage=[]
        capacity=[]
        voltage1=[]
        path_xlsx=os.path.join(path,ls)
        
        df = pd.read_excel(path_xlsx, None)
        a=df['Channel_1-011']
        b=np.asarray(a)
        for i in range(b.shape[0]):
            cycle_index.append(b[i][5])
            voltage.append(b[i][7])
            capacity.append(b[i][9])
            
        for index in np.unique(cycle_index):
            start_index=0
            for temp in range(index):
                start_index=cycle_index.count(temp)+start_index
            capacity1.append(capacity[start_index+cycle_index.count(index)-1]-capacity[start_index])
        ## delete the first charge curve       
            for vol_index in range(start_index+15,start_index+cycle_index.count(index)):
                if voltage[vol_index]<voltage[vol_index-1]:  
                    voltage1.append(voltage[vol_index:start_index+cycle_index.count(index)])
                    break
        ## delete the latter charge curve        
        for item in voltage1:
            for i in range(30,len(item)):
                if i !=len(item)-1:
                    if item[i]>=item[i-1]:
                        voltage2.append(item[:i])
                        break
                else :
                    if item[i]<item[i-1]:
                        voltage2.append(item[:i+1])
                        break
    
    ###some capacity is empty need to be deleted
        if ls=='5.xlsx':
            del capacity1[-1]
            del voltage2[-1] 
        if ls=='7.xlsx':
            del capacity1[-43]
            del voltage2[-43]
            del capacity1[-49]
            del voltage2[-49]
        if ls=='9.xlsx':
            del capacity1[-1] 
        if ls=='10.xlsx':
            del capacity1[-1]
            del voltage2[-1]   
        if ls=='14.xlsx':
            del capacity1[-1]  
        if ls=='18.xlsx':
            del capacity1[-1]
        if ls=='22.xlsx':
            del capacity1[-1]
            del voltage2[-1]     
        if ls=='26.xlsx':
            del capacity1[-1]             
    return voltage2,capacity1
voltage_38,capacity_38=loaddata38(path='path38')



def equallength(data):
    """
    padding the data into a same shape in different battery
    """
    temp=[]
    length=[]
    for item in data:
        length.append(len(item))
    maxlen=np.max(length)
    for i in range(len(data)):
        temp.append(list(np.concatenate((4.2*np.ones((maxlen-len(data[i]),1))[:,0],np.asarray(data[i])))))
    return temp

voltage_35=equallength(voltage_35) 
voltage_36=equallength(voltage_36) 
voltage_37=equallength(voltage_37)   
voltage_38=equallength(voltage_38)    

  
        
def equalall(data1,data2,data3,data4):
    """
    padding the four batteries has a common shape
    """
    temp1=[]
    temp2=[]
    temp3=[]
    temp4=[]
    a=np.max([len(data1[0]),len(data2[0]),len(data3[0]),len(data4[0])])
    for item in data1:
        temp1.append(list(np.concatenate((4.2*np.ones((a-len(item),1))[:,0],np.asarray(item)))))
    for item in data2:
        temp2.append(list(np.concatenate((4.2*np.ones((a-len(item),1))[:,0],np.asarray(item)))))   
    for item in data3:
        temp3.append(list(np.concatenate((4.2*np.ones((a-len(item),1))[:,0],np.asarray(item)))))
    for item in data4:
        temp4.append(list(np.concatenate((4.2*np.ones((a-len(item),1))[:,0],np.asarray(item)))))
    return temp1,temp2,temp3,temp4

voltage_35,voltage_36,voltage_37,voltage_38=equalall(voltage_35,voltage_36,voltage_37,voltage_38)



voltage_35=np.asarray(voltage_35)   
voltage_36=np.asarray(voltage_36)   
voltage_37=np.asarray(voltage_37)      
voltage_38=np.asarray(voltage_38)  











    

