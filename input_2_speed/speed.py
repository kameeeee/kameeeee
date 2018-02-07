# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import RNN_model
from chainer import Variable, optimizers, serializers, cuda
import chainer.functions as F

def rmse(predict, true):
    return np.sqrt(((true - predict)**2).mean())

def make_minibatch(N, delta, Minibatch_size, SpeedData, BrakeData):
    rnd = np.random.rand(Minibatch_size)*(N-delta-1)
    source = []
    target = []
    for r in rnd:
        s = [ [ SpeedData[v], BrakeData[v] ] for v in range(int(r),int(r)+delta) ]
        t = [ [ SpeedData[v] ] for v in range(int(r+1),int(r)+delta+1) ]
        source.append(s)
        target.append(t)
    source = np.array(source)
    target = np.array(target)
    source = source.transpose(1,0,2)
    target = target.transpose(1,0,2)
    return source,target

'''
fig = plt.figure(1)
plt.subplot(221)
plt.title('speed')
plt.plot(xrange(SpeedData.size), SpeedData, '.')
plt.subplot(222)
plt.title('brake')
plt.plot(xrange(BrakeData.size), BrakeData, '.')
plt.subplot(223)
plt.title('accelator')
plt.plot(xrange(AccelatorData.size), AccelatorData, '.')
plt.subplot(224)
plt.title('RPM')
plt.plot(xrange(RPMData.size), RPMData, '.')
plt.show()
'''

#CSV read
data = np.loadtxt('/home/kame2/workspace/candata/log/taniguchi/fit_all_data.csv', delimiter=',', skiprows=1)  

#Data set
SpeedData = np.array(data).T[0,:] #cycle 10ms
BrakeData = np.array(data).T[1,:] #cycle 10ms

#fit cycle
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 50)] )
BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, 50)] )

gpu_flag = 0
xp = np

if gpu_flag >= 0:
    cuda.check_cuda_available()
    xp = cuda.cupy
    
#parameter
delta = 60
Minibatch_size = int(SpeedData.size/delta)
loss_val = 100
epoch = 0
n_epoch = 5000

network = "7_Linear1_100_LSTM1_100_Linear1_100_interval_50_delta"
print "network: {}  delta: {}  starts!".format(network, delta)

model = RNN_model.RNN_model()

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()
    
optimizer = optimizers.Adam()
optimizer.setup(model)
line_x = []
line_y = []
line_x_val = []
line_y_val = []
RMSE = []
now_loss = []

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
plt.title('loss')
ax2 = fig.add_subplot(212)

plt.title('validation')
plt.tight_layout()
ax2.plot(xrange(SpeedData[3950:].size-1), SpeedData[3951:])

while epoch < n_epoch:
    [source, target] = make_minibatch(SpeedData[:3949].size, delta, Minibatch_size, SpeedData[:3949], BrakeData[:3949])
    model.reset_state()
    model.zerograds()
    loss = 0
    
    for i in range(0,delta-1):
        s = Variable(xp.asarray(source[i], dtype=xp.float32))
        t = Variable(xp.asarray(target[i], dtype=xp.float32))
        loss += model(s,t)
        
    loss.backward()
    optimizer.update()
    
    if epoch % 100 == 0:
        loss_val = cuda.to_cpu(loss.data)/delta
        line_x.append(epoch)
        line_y.append(loss_val)
        val_flag = 0
        
        if epoch % 100 == 0:
            val_flag = 1
            model.reset_state()
            model.zerograds()
            validation_loss = 0
            answer_speed = []
            
            for i in xrange(SpeedData[3950:].size-1):
                s = Variable( xp.asarray( [ [ SpeedData[i+3950], BrakeData[i+3950] ] ], dtype = xp.float32 ) )
                t = Variable( xp.asarray( [ [ SpeedData[i+3951] ] ], dtype = xp.float32 ) )
                ans = model.fwd(s)
                
                if gpu_flag >= 0:
                    loss = F.mean_squared_error(ans, t)
                    answer_speed.append( cuda.to_cpu (ans.data[0][0] ) )
                    validation_loss += cuda.to_cpu( loss.data )
                else:
                    answer_speed.append( cuda.to_cpu (ans.data[0][0] ) )
                    loss = F.mean_squared_error(ans, t)
                    validation_loss += loss.data
                    
            validation_loss /= SpeedData[3950:].size
            answer_speed = np.array(answer_speed)
            line_y_val.append(validation_loss)
            line_x_val.append(epoch)
            error = rmse(answer_speed, SpeedData[3951:])
            
        if epoch == 0:
            lines, = ax1.plot(line_x, line_y)
            lines2, = ax1.plot(line_x_val, line_y_val)
            lines3, = ax2.plot(xrange(SpeedData[3950:].size-1), answer_speed)
            
        else:
            lines.set_data(line_x, line_y)
            if val_flag == 1:
                lines2.set_data(line_x_val, line_y_val)
                lines3.set_data(xrange(SpeedData[3950:].size-1), answer_speed)
            ax1.set_xlim((0,epoch))
            ax1.set_ylim((0,np.max(line_y)))
            plt.pause(.01)
            
            print "epoch: {}/{} now_loss : {} validatiom_loss : {} RMSE : {}".format(epoch,n_epoch,loss_val,validation_loss,error)

            #RMSE.append(error)
            #now_loss.append(loss_val)
            
    epoch += 1

print "learning finish!"
print ""

plt.savefig("./png/7_Linear1_100_LSTM1_100_Linear1_100_interval_50_delta60.png")

if gpu_flag >= 0:
    model.to_cpu()

serializers.save_npz("./learned/7_Linear1_100_LSTM1_100_Linear1_100_interval_50_delta60.model", model)
serializers.save_npz("./learned/7_Linear1_100_LSTM1_100_Linear1_100_interval_50_delta60.state", optimizer)
