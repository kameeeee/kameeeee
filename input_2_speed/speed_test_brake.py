
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
data1 = np.loadtxt('/home/kame2/workspace/candata/log/taniguchi/speed_brake_fullver.csv', delimiter=',', skiprows=1)               
#data2 = np.loadtxt('/home/kame2/workspace/candata/log/accelator_rpm_freq.csv', delimiter=',', skiprows=1) 

#Data set
SpeedData = np.array(data1).T[0,:462275] #cycle 10ms
BrakeData = np.array(data1).T[1,:462275] #cycle 10ms
#AccelatorData = np.array(data2).T[1,:6960] #cycle 16ms 
#RPMData = np.array(data2).T[2,:6960] #cycle 16ms

#SpeedData[3000:3200] = 40
#SpeedData[5300:5500] = 50

#fit cycle
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 41)] )
BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, 41)] )
#AccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, 25)] )
#RPMData = np.array( [RPMData[v] for v in xrange(0, RPMData.size, 25)] )

#学習済みモデルのロード

model = RNN_model.RNN_model()
serializers.load_npz("./learned/epoch5000/Linear1_100_LSTM1_100_Linear1_100_delta60.model", model)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
plt.title('validation')
plt.tight_layout()

model.reset_state()
model.zerograds()

answer_speed = []
answer = -1
flag = -1
initial = 0
#nextdata = rand(3) * 2 + 32
nextdata = [1, 10]
attacker = []

for i in xrange(SpeedData.size-1):
    
    delta_pre = []
    attacker.append(nextdata[0])
    nextdata[1] = BrakeData[i]
    
    if answer == -1:
        s = Variable( np.asarray( [ [ SpeedData[i], BrakeData[i] ] ], dtype=np.float32) )
    else:
        if initial == 0:
            s1 = Variable( np.asarray( [ [ SpeedData[i], nextdata[0] ] ], dtype=np.float32) )
            s2 = Variable( np.asarray( [ [ SpeedData[i], nextdata[1] ] ], dtype=np.float32) )
            
            pre1 = model.fwd(s1).data[0][0]
            pre2 = model.fwd(s2).data[0][0]
            
            delta_pre.append( mse(pre1, SpeedData[i+1]) )
            delta_pre.append( mse(pre2, SpeedData[i+1]) )
            
            arg = np.argmin(delta_pre)
            answer_brake = nextdata[arg]
            initial = 1
    
    if flag == 1:
        nextdata[0] += 0.5
        if nextdata[0] > 50:
            flag = 0
    else:
        nextdata[0] -= 0.5
        if nextdata[0] < 1:    
            flag = 1
                    
    #print "answer : {}".format(answer)
    
attacker.append(nextdata[0])
attacker = np.array(attacker)
answer_speed = np.array(answer_speed)

error = rmse(answer_speed, SpeedData[1:])
print "RMSE : {}".format(error)

ax1.plot(xrange(SpeedData.size-1), SpeedData[1:], '--')
ax1.plot(xrange(SpeedData.size-1), answer_speed)
ax1.plot(xrange(SpeedData.size-1), attacker[1:], '-.')
plt.show()

