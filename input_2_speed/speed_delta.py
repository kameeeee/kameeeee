# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import RNN_model
from chainer import Variable, optimizers, serializers, cuda
import chainer.functions as F

def rmse(predict, true):
    return np.sqrt(((true - predict)**2).mean())

def make_minibatch(N, delta, Minibatch_size, SpeedData, RPMData):
    rnd = np.random.rand(Minibatch_size)*(N-delta-1)
    source = []
    target = []
    for r in rnd:
        s = [ [ SpeedData[v], RPMData[v] ] for v in range(int(r),int(r)+delta) ]
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

count = 0

delta = 30

while count < 3:

    #CSV read
    data1 = np.loadtxt('/home/kame2/workspace/candata/log/speed_brake_fullver.csv', delimiter=',', skiprows=1)  
    data2 = np.loadtxt('/home/kame2/workspace/candata/log/accelator_engine_fullver.csv', delimiter=',', skiprows=1) 
    
    #Data set
    SpeedData = np.array(data1).T[0,:462275] #cycle 10ms
    #BrakeData = np.array(data1).T[1,:462275] #cycle 10ms
    #AccelatorData = np.array(data2).T[0,:281875] #cycle 16ms 
    RPMData = np.array(data2).T[1,:281875] #cycle 16ms
    
    #fit cycle
    SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, 41)] )
    #BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, 41)] )
    #AccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, 25)] )
    RPMData = np.array( [RPMData[v] for v in xrange(0, RPMData.size, 25)] )
    
    
    gpu_flag = 0
    xp = np
    
    if gpu_flag >= 0:
        cuda.check_cuda_available()
        xp = cuda.cupy
        
    #parameter
    Minibatch_size = int(SpeedData.size/delta)
    loss_val = 100
    epoch = 0
    n_epoch = 5000
    
    network = "Linear1_50_LSTM1_50_Linear1_50_delta"
    
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
    ax2.plot(xrange(SpeedData.size-1), SpeedData[1:])
    
    while epoch < n_epoch:
        [source, target] = make_minibatch(SpeedData.size, delta, Minibatch_size, SpeedData, RPMData)
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
                
                for i in xrange(SpeedData.size-1):
                    s = Variable( xp.asarray( [ [ SpeedData[i], RPMData[i] ] ], dtype = xp.float32 ) )
                    t = Variable( xp.asarray( [ [ SpeedData[i+1] ] ], dtype = xp.float32 ) )
                    ans = model.fwd(s)
                    
                    if gpu_flag >= 0:
                        loss = F.mean_squared_error(ans, t)
                        answer_speed.append( cuda.to_cpu (ans.data[0][0] ) )
                        validation_loss += cuda.to_cpu( loss.data )
                    else:
                        answer_speed.append( cuda.to_cpu (ans.data[0][0] ) )
                        loss = F.mean_squared_error(ans, t)
                        validation_loss += loss.data
                        
                validation_loss /= SpeedData.size
                answer_speed = np.array(answer_speed)
                line_y_val.append(validation_loss)
                line_x_val.append(epoch)
                error = rmse(answer_speed, SpeedData[1:])
                
            if epoch == 0:
                lines, = ax1.plot(line_x, line_y)
                lines2, = ax1.plot(line_x_val, line_y_val)
                lines3, = ax2.plot(xrange(SpeedData.size-1), answer_speed)
                
            else:
                lines.set_data(line_x, line_y)
                if val_flag == 1:
                    lines2.set_data(line_x_val, line_y_val)
                    lines3.set_data(xrange(SpeedData.size-1), answer_speed)
                ax1.set_xlim((0,epoch))
                ax1.set_ylim((0,np.max(line_y)))
                plt.pause(.01)
                
                print "epoch: {}/{} now_loss : {} validatiom_loss : {} RMSE : {}".format(epoch,n_epoch,loss_val,validation_loss,error)
    
                RMSE.append(error)
                now_loss.append(loss_val)
                
        epoch += 1
    
    print "learning finish!"
    print ""
    
    if count == 0:
        plt.savefig("./png/Linear1_50_LSTM1_50_Linear1_50_delta30.png")

    if count == 1:
        plt.savefig("./png/Linear1_50_LSTM1_50_Linear1_50_delta60.png")
    
    else:
        plt.savefig("./png/Linear1_50_LSTM1_50_Linear1_50_delta90.png")
            
    if gpu_flag >= 0:
        model.to_cpu()
    
    if count == 0:
        serializers.save_npz("./learned/Linear1_50_LSTM1_50_Linear1_50_delta30.model", model)
        serializers.save_npz("./learned/Linear1_50_LSTM1_50_Linear1_50_delta30.state", optimizer)

    if count == 1:
        serializers.save_npz("./learned/Linear1_50_LSTM1_50_Linear1_50_delta60.model", model)
        serializers.save_npz("./learned/Linear1_50_LSTM1_50_Linear1_50_delta60.state", optimizer)

    else:
        serializers.save_npz("./learned/Linear1_50_LSTM1_50_Linear1_50_delta90.model", model)
        serializers.save_npz("./learned/Linear1_50_LSTM1_50_Linear1_50_delta90.state", optimizer)
    
    plt.clf()
    plt.close()
    
    count += 1
    delta += 30
    
print "learning finish!"
print ""
