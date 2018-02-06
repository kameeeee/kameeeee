
# -*- coding: utf-8 -*-
import numpy as np
from chainer import Variable, optimizers, serializers
import matplotlib.pyplot as plt
from numpy.random import *
import chainer.functions as F

#自作学習モデルをロード
import RNN_model

def rmse(predict, true):
    return np.sqrt(((true - predict)**2).mean())

def mse(predict, true):
    return (true - predict)**2

#CSV read
data = np.loadtxt('/home/kame2/workspace/candata/log/taniguchi/fit_all_data.csv', delimiter=',', skiprows=1)               
#data2 = np.loadtxt('/home/kame2/workspace/candata/log/accelator_rpm_freq.csv', delimiter=',', skiprows=1) 

#Data set
SpeedData = np.array(data).T[0,:] #cycle 10ms
BrakeData = np.array(data).T[1,:] #cycle 10ms
AccelatorData = np.array(data).T[2,:] #cycle 16ms 
#RPMData = np.array(data2).T[2,:6960] #cycle 16ms

#SpeedData[3000:3200] = 40
#SpeedData[5300:5500] = 50

#fit cycle
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 50)] )
BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, 50)] )
AccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, 50)] )
#RPMData = np.array( [RPMData[v] for v in xrange(0, RPMData.size, 25)] )

#学習済みモデルのロード
model = RNN_model.RNN_model()
serializers.load_npz("./learned/Linear1_100_LSTM2_100_Linear1_100_delta300.model", model)

model.reset_state()
model.zerograds()

answer_speed = []
answer = -1
flag = -1
#nextdata = rand(3) * 2 + 32
nextdata = [1, 10]
attacker = []

for i in xrange(SpeedData.size-1):
    
    delta_pre = []
    attacker.append(nextdata[0])
    
    if answer == -1:
        s = Variable( np.asarray( [ [ SpeedData[i], BrakeData[i], AccelatorData[i] ] ], dtype=np.float32) )
    else:
        s = Variable( np.asarray( [ [ answer, BrakeData[i], AccelatorData[i] ] ], dtype=np.float32) )
        
    pre = model.fwd(s).data[0][0]
    nextdata[1] = SpeedData[i+1]
    
    for n in nextdata:
        delta_pre.append( mse(pre, n) )
        
    
    arg = np.argmin(delta_pre)
    answer = nextdata[arg]
    answer_speed.append( answer )
    
    if flag == 1:
        nextdata[0] += 1
        if nextdata[0] > 50:
            flag = 0
    else:
        nextdata[0] -= 1
        if nextdata[0] < 1:    
            flag = 1
                    
    #print "answer : {}".format(answer)
    
attacker.append(nextdata[0])
attacker = np.array(attacker)
answer_speed = np.array(answer_speed)

error = rmse(answer_speed, SpeedData[1:])
print "RMSE : {}".format(error)


fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.title('validation', fontsize=22)
#plt.xlabel('Number of Frames', fontsize=22)
plt.ylabel('speed(km/h)', fontsize=22)
plt.plot(answer_speed[320:720], 'k', label='output')
plt.plot(attacker[320:720], ':r', label='attacker')
plt.plot(SpeedData[321:721], ':b', label='speed data')
plt.legend(loc='upper right', fontsize=16)
plt.tick_params(labelsize=18)
ax2 = fig.add_subplot(212)
plt.title('brake and accelator data', fontsize=22)
plt.xlabel('Number of Frames', fontsize=22)
plt.ylabel('brake amount', fontsize=22)
ax2.plot(BrakeData[321:721], 'g', label='brake amount')
plt.legend(loc='upper left',fontsize=16)
plt.tick_params(labelsize=18)
ax3 = ax2.twinx()  # 2つのプロットを関連付ける
ax3.plot(AccelatorData[321:721], 'm', label='accelator amount')
plt.ylabel('accelator amount', fontsize=22)
plt.legend(loc='upper right', fontsize=16)
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()


'''
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.title('output')
plt.ylabel('speed(km/h)')
plt.plot(answer_speed, 'b', label='output')
plt.plot(attacker, 'r' '--')
plt.plot(SpeedData, '-.')
ax2 = fig.add_subplot(212)
plt.title('test data')
plt.xlabel('time')
plt.ylabel('brake amount')
ax2.plot(BrakeData, 'g', label='test data(brake amount)')
ax3 = ax2.twinx()  # 2つのプロットを関連付ける
ax3.plot(AccelatorData, 'm', label='test data(accelator amount)')
plt.ylabel('accelator amount')
plt.show()
'''

'''
fig, ax1 = plt.subplots()
plt.title('test data(speed)')
plt.xlabel('time')
plt.ylabel('speed(km/h)')
ax1.plot(SpeedData[321:721], 'b', label='test data(speed)')
fig, ax2 = plt.subplots()
plt.title('output')
plt.xlabel('time')
plt.ylabel('speed(km/h)')
ax2.plot(answer_speed[320:720], 'm', label='output')
fig, ax3 = plt.subplots()
plt.title('attack data')
plt.xlabel('time')
plt.ylabel('speed(km/h)')
ax3.plot(attacker[321:721], 'r', label='attack data')
plt.show()
'''

'''
fig = plt.figure()
ax1 = fig.add_subplot(311)
plt.title('speed')
plt.ylabel('speed(km/h)')
ax1.plot(SpeedData[321:721], 'b')
ax2 = fig.add_subplot(312)
plt.title('brake')
plt.ylabel('brake amount')
ax2.plot(BrakeData[321:721], 'g')
ax3 = fig.add_subplot(313)
ax3.plot(AccelatorData[321:721], 'm')
plt.title('accelator')
plt.xlabel('time')
plt.ylabel('accelator amount')
plt.tight_layout()
plt.show()
'''
