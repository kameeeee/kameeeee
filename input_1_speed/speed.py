# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import RNN_model
from chainer import Variable, optimizers, serializers, cuda
import chainer.functions as F

def rmse(predict, true):
    return np.sqrt(((true - predict)**2).mean())

def make_minibatch(N, delta, Minibatch_size, SpeedData):
    rnd = np.random.rand(Minibatch_size)*(N-delta-1)
    source = []
    target = []
    for r in rnd:
        s = [ [ SpeedData[v] ] for v in range(int(r),int(r)+delta)]
        t = [ [ SpeedData[v] ] for v in range(int(r+1),int(r)+delta+1)]
        source.append(s)
        target.append(t)
    source = np.array(source)
    target = np.array(target)
    source = source.transpose(1,0,2)
    target = target.transpose(1,0,2)
    return source, target

#CSV read
data = np.loadtxt('/home/kame2/workspace/candata/log/taniguchi/fit_all_data.csv', delimiter=',', skiprows=1)

#Data set
SpeedData = np.array(data).T[0,:]
SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 50)], dtype=np.float32 )/100.0

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
#serializers.load_npz("./learned/epoch5000/Linear1_100_LSTM1_100_Linear1_100_delta30.model", model)

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
    [source, target] = make_minibatch(SpeedData[:3949].size, delta, Minibatch_size, SpeedData[:3949])
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
        
        #誤差のプロット
        line_x.append(epoch)
        line_y.append(loss_val)
        
        val_flag = 0
        
        if epoch % 100 == 0:
            val_flag = 1
            
            #validation
            model.reset_state()
            model.zerograds()
            validation_loss = 0
            answer = []
            
            for i in xrange(SpeedData[3950:].size-1):
                s = Variable( xp.asarray( [ [ SpeedData[i+3950] ] ], dtype=xp.float32) )
                t = Variable( xp.asarray( [ [ SpeedData[i+3951] ] ], dtype=xp.float32) )
                ans = model.fwd(s)
                
                if gpu_flag >= 0:
                    loss = F.mean_squared_error(ans, t)
                    answer.append( cuda.to_cpu (ans.data[0][0] ) )
                    validation_loss += cuda.to_cpu( loss.data )
                else:
                    answer.append( ans.data[0][0] )
                    loss = F.mean_squared_error(ans, t)
                    validation_loss += loss.data
                    
            validation_loss /= SpeedData[3950:].size
            answer = np.array(answer)
            line_y_val.append(validation_loss)
            line_x_val.append(epoch)
            error = rmse(answer, SpeedData[3951:])
        
        if epoch == 0:
            lines, = ax1.plot(line_x, line_y)
            lines2, = ax1.plot(line_x_val, line_y_val)
            lines3, = ax2.plot(xrange(SpeedData[3950:].size-1), answer)
            
        else:
            lines.set_data(line_x, line_y)
            if val_flag == 1:
                lines2.set_data(line_x_val, line_y_val)
                lines3.set_data(xrange(SpeedData[3950:].size-1), answer)
            ax1.set_xlim((0,epoch))
            ax1.set_ylim((0,np.max(line_y)))
            plt.pause(.01)
            
            print "epoch: {}/{} now_loss : {} validatiom_loss : {} RMSE : {}".format(epoch,n_epoch,loss_val,validation_loss,error)
        
    epoch += 1

print "learning finish!"
print ""

plt.savefig("./png/Linear1_100_LSTM1_100_Linear1_100_interval_50_delta60_2.png")

#学習済みモデルの保存
if gpu_flag >= 0:
    model.to_cpu()

serializers.save_npz("./learned/Linear1_100_LSTM1_100_Linear1_100_interval_50_delta60.model", model)
serializers.save_npz("./learned/Linear1_100_LSTM1_100_Linear1_100_interval_50_delta60.state", optimizer)