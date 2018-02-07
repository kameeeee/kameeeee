# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
    
def mse(predict, true):
    a = predict - true
    return a
    
#CSV read
data1 = np.loadtxt('./same_route/yoshida/return/speed_brake.csv', delimiter=',', skiprows=1)  
data2 = np.loadtxt('./same_route/yoshida/return/accelator_rpm.csv', delimiter=',', skiprows=1) 

#Data set
SpeedData_time = np.array(data1).T[0,:] #cycle 10ms
SpeedData = np.array(data1).T[1,:]
#BrakeData = np.array(data1).T[2,:] #cycle 10ms
AccelatorData_time = np.array(data2).T[0,:]
AccelatorData = np.array(data2).T[1,:] #cycle 16ms 
#RPMData = np.array(data2).T[2,:] #cycle 16ms

fit_brake = []
count = 0
cnt = 0
fin_idx = 0

for n_idx, n in enumerate(AccelatorData_time):
    
    for i_idx, i in enumerate(SpeedData_time[fin_idx:]):
        
        fit_time = mse(n, i)
        
        if fit_time < 0:
            
            if abs( mse(AccelatorData_time[n_idx-1], SpeedData_time[i_idx-1]) ) < abs(fit_time):
                fit_brake.append(SpeedData[fin_idx+i_idx-1])
                break
                
            else:
                fit_brake.append(SpeedData[fin_idx+i_idx])
                break
                
    fin_idx += i_idx

fit_brake = np.array(fit_brake)

print "Accelator_size: {}  fit_data_size: {}".format(AccelatorData.size, fit_brake.size)

with open('./same_route/yoshida/return/fit_speed.csv', 'w') as f:
    writer = csv.writer(f)  # writerオブジェクトを作成
    for fs in fit_brake:
        writer.writerow([fs])  # 内容を書き込む

fig, ax1 = plt.subplots()
plt.title('speed_accelator')
ax1.plot(fit_brake, '--r')
ax2 = ax1.twinx()  # 2つのプロットを関連付ける
ax2.plot(AccelatorData,'--k')
plt.show()

