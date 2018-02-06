
# -*- coding: utf-8 -*-
import numpy as np
from chainer import Variable, optimizers, serializers
import matplotlib.pyplot as plt

#自作学習モデルをロード
import RNN_model

data = np.loadtxt('/home/kame2/workspace/candata/log/speed_brake_freq.csv', delimiter=',', skiprows=1)               
#SpeedData = data
SpeedData = np.array(data).T[0,:]

SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 25)] )

#SpeedData[3000:3200] = 40
#SpeedData[5300:5500] = 50

#学習済みモデルのロード
model = RNN_model.RNN_model()
serializers.load_npz("./learned/epoch5000/Linear1_50_LSTM1_50_Linear1_50_delta60.model", model)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
plt.title('validation')
plt.xlabel('time')
plt.ylabel('speed(km/h)')

model.reset_state()
model.zerograds()
answer = []
for i in xrange(SpeedData.size-1):
    s = Variable( np.asarray( [ [ SpeedData[i] ] ], dtype=np.float32) )
    ans = model.fwd(s)
    answer.append( ans.data[0][0] )
answer = np.array(answer)
    

ax1.plot(xrange(SpeedData.size-1), SpeedData[1:], '--')
ax1.plot(xrange(SpeedData.size-1), answer)
plt.show()