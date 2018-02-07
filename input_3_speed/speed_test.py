
# -*- coding: utf-8 -*-
import numpy as np
from chainer import Variable, optimizers, serializers
import matplotlib.pyplot as plt

#自作学習モデルをロード
import RNN_model

def rmse(predict, true):
    return np.sqrt(((true - predict)**2).mean())

#CSV read
data = np.loadtxt('/home/kame2/workspace/model/log/unknow_data.csv', delimiter=',', skiprows=1)               
#data2 = np.loadtxt('/home/kame2/workspace/candata/log/accelator_engine_rpm_fullver.csv', delimiter=',', skiprows=1) 

#Data set
SpeedData = np.array(data).T[0,:63500] #cycle 10ms
BrakeData = np.array(data).T[1,:63500] #cycle 10ms
AccelatorData = np.array(data).T[2,:63500] #cycle 16ms
#RPMData = np.array(data2).T[1,:281875] #cycle 16ms

#fit cycle
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 1)] )/100
BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, 1)] )/100
AccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, 1)] )/100
#RPMData = np.array( [RPMData[v] for v in xrange(0, RPMData.size, 25)] )

#SpeedData[3000:3200] = 0
#SpeedData[5300:5500] = 50

#学習済みモデルのロード
model = RNN_model.RNN_model()
serializers.load_npz("/home/kame2/workspace/model/defore_dataset_learned/Linear_1_LSTM_1_Linear_1/50node_3/50node_3.model", model)

model.reset_state()
model.zerograds()
answer_speed = []

for i in xrange(SpeedData.size-1):
    s = Variable( np.asarray( [ [ SpeedData[i], BrakeData[i], AccelatorData[i] ] ], dtype=np.float32) )
    ans = model.fwd(s)
    answer_speed.append( ans.data[0][0] )
    
answer_speed = np.array(answer_speed)
error = rmse(answer_speed, SpeedData[1:])

print "RMSE : {}".format(error)

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('time (minutes)', fontsize=22)
plt.ylabel('speed(km/h)', fontsize=22)
plt.plot(SpeedData[1:]*100, label = "predict data")
plt.plot(answer_speed*100, label = "crrect data")
plt.xticks([0,12433,24866,37300,49733,62167], ["0","4","8","12","16","20"])
plt.legend(loc='upper right', fontsize=22)
plt.tick_params(labelsize=22)
plt.show()
