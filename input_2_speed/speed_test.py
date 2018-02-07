
# -*- coding: utf-8 -*-
import numpy as np
from chainer import Variable, optimizers, serializers
import matplotlib.pyplot as plt

#自作学習モデルをロード
import RNN_model

def rmse(predict, true):
    return np.sqrt(((true - predict)**2).mean())

#CSV read
data1 = np.loadtxt('/home/kame2/workspace/candata/log/speed_brake_fullver.csv', delimiter=',', skiprows=1)               
#data2 = np.loadtxt('/home/kame2/workspace/candata/log/accelator_engine_rpm_fullver.csv', delimiter=',', skiprows=1) 

#Data set
SpeedData = np.array(data1).T[0,:462275] #cycle 10ms
BrakeData = np.array(data1).T[1,:462275] #cycle 10ms
#AccelatorData = np.array(data2).T[0,:281875] #cycle 16ms
#RPMData = np.array(data2).T[1,:281875] #cycle 16ms

#fit cycle
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 41)] )
BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, 41)] )
#AccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, 25)] )
#RPMData = np.array( [RPMData[v] for v in xrange(0, RPMData.size, 25)] )

#SpeedData[3000:3200] = 0
#SpeedData[5300:5500] = 50

#学習済みモデルのロード
model = RNN_model.RNN_model()
serializers.load_npz("./learned/epoch5000/Linear1_100_LSTM1_100_Linear1_100_delta90.model", model)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
plt.title('validation')
plt.tight_layout()

model.reset_state()
model.zerograds()
answer_speed = []

for i in xrange(SpeedData.size-1):
    s = Variable( np.asarray( [ [ SpeedData[i], BrakeData[i] ] ], dtype=np.float32) )
    ans = model.fwd(s)
    answer_speed.append( ans.data[0][0] )
    
answer_speed = np.array(answer_speed)
error = rmse(answer_speed, SpeedData[1:])

print "RMSE : {}".format(error)

ax1.plot(xrange(SpeedData.size-1), SpeedData[1:], '--')
ax1.plot(xrange(SpeedData.size-1), answer_speed)
plt.show()