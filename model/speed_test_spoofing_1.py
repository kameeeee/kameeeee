
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
data = np.loadtxt('./unknow_data.csv', delimiter=',', skiprows=1)               

#Data set
SpeedData = np.array(data).T[0,:] #cycle 10ms

#fit cycle
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 50)] )/100

SpeedData = SpeedData[:1100]

#学習済みモデルのロード
model = RNN_model.RNN_model()
serializers.load_npz("./Linear_1_LSTM_1_Linear_1/50node_1/50node_1.model", model)

model.reset_state()
model.zerograds()

sp=np.array([])

Input = 1

if sp.size == 0:
    flag = -1
    i = 0
    s = []
    for c in range(SpeedData.size):
        if flag == 1:
            i += 1
            s.append(i)
            if i > 50:
                flag = -1
        else:
            i -= 1
            s.append(i)
            if i < 1:    
                flag = 1
    sp = np.array(s)
sp = np.asarray(sp, dtype=np.float32)
sp /= 100
answer_speed = []
answer = -1
nextdata = [1, 10]

for i in xrange(SpeedData.size-1):
    
    delta_pre = []
    
    if answer == -1:
        if Input == 1:
            s = Variable( np.asarray( [ [SpeedData[i] ] ], dtype=np.float32) )
        if Input == 2:
            s = Variable( np.asarray( [ [SpeedData[i],BrakeData[i] ] ], dtype=np.float32) )
        if Input == 3:
            s = Variable( np.asarray( [ [SpeedData[i],BrakeData[i],AccelatorData[i] ] ], dtype=np.float32) )
    else:
        if Input == 1:
            s = Variable( np.asarray( [ [ answer ] ], dtype=np.float32) )
        if Input == 2:
            s = Variable( np.asarray( [ [ answer,BrakeData[i] ] ], dtype=np.float32) )
        if Input == 3:
            s = Variable( np.asarray( [ [ answer,BrakeData[i],AccelatorData[i] ] ], dtype=np.float32) )
        
    pre = model.fwd(s).data[0][0]
    nextdata[1] =SpeedData[i+1]
    
    for n in nextdata:
        delta_pre.append( rmse(pre, n) )
        
    arg = np.argmin(delta_pre)
    answer = nextdata[arg]
    answer_speed.append( answer )
    
    nextdata[0] = sp[i]

answer_speed = np.array(answer_speed)
error = rmse(answer_speed, SpeedData[1:])*100
print "TestSP RMSE : {}".format(error)

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.title('validation', fontsize=22)
plt.xlabel('Number of Frames', fontsize=22)
plt.ylabel('speed(km/h)', fontsize=22)
plt.plot(answer_speed*100, 'k', label='output')
plt.plot(sp[1:]*100, ':r', label='attacker')
plt.plot(SpeedData*100, ':b', label='speed data')
plt.legend(loc='upper right', fontsize=16)
plt.tick_params(labelsize=18)
plt.show()
