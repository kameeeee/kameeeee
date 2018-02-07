# -*- coding: utf-8 -*-


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
#BrakeData = np.array(data).T[1,:] #cycle 10ms
#AccelatorData = np.array(data).T[2,:] #cycle 16ms 
#RPMData = np.array(data2).T[2,:6960] #cycle 16ms

#SpeedData[3000:3200] = 40
#SpeedData[5300:5500] = 50

#fit cycle
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 50)] )
#BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, 25)] )
#AccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, 25)] )
#RPMData = np.array( [RPMData[v] for v in xrange(0, RPMData.size, 25)] )

#学習済みモデルのロード

model = RNN_model.RNN_model()
serializers.load_npz("./learned/Linear1_100_LSTM1_100_Linear1_100_interval_50_delta60.model", model)

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
        s = Variable( np.asarray( [ [ SpeedData[i] ] ], dtype=np.float32) )
    else:
        s = Variable( np.asarray( [ [ answer ] ], dtype=np.float32) )
        
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

'''
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.title('attacker data', fontsize=22)
plt.xlabel('Number of Frames', fontsize=22)
plt.ylabel('speed(km/h)', fontsize=22)
plt.plot(attacker[320:720], 'r', label='attacker')
plt.legend(loc='upper right', fontsize=16)
plt.tick_params(labelsize=18)
plt.show()
'''

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.title('validation', fontsize=22)
plt.xlabel('Number of Frames', fontsize=22)
plt.ylabel('speed(km/h)', fontsize=22)
plt.plot(answer_speed[320:720], 'k', label='output')
plt.plot(attacker[320:720], ':r', label='attacker')
plt.plot(SpeedData[321:721], ':b', label='speed data')
plt.legend(loc='upper right', fontsize=16)
plt.tick_params(labelsize=18)
plt.show()


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
ax3.plot(attacker[320:720], 'r', label='attack data')
plt.show()
'''

