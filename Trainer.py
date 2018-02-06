# -*- coding: utf-8 -*-
import numpy as np
import RNN_model1
from chainer import Variable, optimizers, serializers, cuda
import chainer.functions as F
import matplotlib.pyplot as plt

import TrainFigure

import fileSystem
import os

class Trainer():
    def __init__(self, Train):
        self._Train = Train
    
    def TrainSetup(self, inputFileName, Name, model, thin=50, Norm=[1, 1, 1], delta=60, target_epoch=500, Interrupt=100, TestInputFile=""):
        self._Name = Name
        self._Norm = Norm
        self._delta = delta
        self._target_epoch = target_epoch
        self._Interrupt = Interrupt      
        self._gpu_flag = 0
        self._xp = np
        self._thin = thin
        if len(TestInputFile) == 0:
            self._TestInputFile = inputFileName
        else:
            self._TestInputFile = TestInputFile
        
        if self._gpu_flag >= 0:
            cuda.check_cuda_available()
            self._xp = cuda.cupy
        
        self._model = model
        
        if self._gpu_flag >= 0:
            cuda.get_device(self._gpu_flag).use()
            self._model.to_gpu()
            
        self._optimizer = optimizers.Adam()
        self._optimizer.setup(self._model)
        
        #CSV read
        data = np.loadtxt(inputFileName, delimiter=',', skiprows=1)
        
        fileSystem.MakeFolder(os.path.join("./", Name))
        
        self._Input = self._model.InputNum()
        
        #Data set
        SpeedData = np.array(data).T[0,:]
        if self._Input >= 2:
            BrakeData = np.array(data).T[1,:]
        if self._Input == 3:
            AccelatorData = np.array(data).T[2,:]
        '''
        if self._thin == 0:
            self._SpeedData = np.array(SpeedData)/self._Norm[0]
            self._BrakeData = np.array(BrakeData)/self._Norm[1]
            self._AccelatorData = np.array(AccelatorData)/self._Norm[2]
        else:
        '''
        self._SpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, self._thin)], dtype=np.float32 )/self._Norm[0]
        if self._Input >= 2:
            self._BrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, self._thin)], dtype=np.float32 )/self._Norm[1]
        if self._Input == 3:
            self._AccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, self._thin)], dtype=np.float32 )/self._Norm[2]
    
        self.TestSetup(self._TestInputFile, self._model, thin=self._thin, Norm=self._Norm)
        
        self._fig = TrainFigure.TrainFigure(True, [np.arange(self._TestSpeedData.size-1), self._TestSpeedData[1:]*self._Norm[0]])

    def TestSetup(self, inputFileName, model, thin=50, Norm=[1, 1, 1]):
        self._model = model
        self._Input = self._model.InputNum()
        self._Norm = Norm
        self._thin = thin
        data = np.loadtxt(inputFileName, delimiter=',', skiprows=1)
        
        #Data set
        SpeedData = np.array(data).T[0,:]
        if self._Input >= 2:
            BrakeData = np.array(data).T[1,:]
        if self._Input == 3:
            AccelatorData = np.array(data).T[2,:]
        '''    
        if self._thin == 0:
            self._SpeedData = SpeedData
            self._Brakedata = BrakeData
            self._AccelatorData = AccelatorData
        else:
        '''                
        self._TestSpeedData = np.array( [SpeedData[v] for v in xrange(0, SpeedData.size, thin)], dtype=np.float32 )/self._Norm[0]
        if self._Input >= 2:
            self._TestBrakeData = np.array( [BrakeData[v] for v in xrange(0, BrakeData.size, thin)], dtype=np.float32 )/self._Norm[1]
        if self._Input == 3:
            self._TestAccelatorData = np.array( [AccelatorData[v] for v in xrange(0, AccelatorData.size, thin)], dtype=np.float32 )/self._Norm[2]

    def rmse(self, predict, true):
        return np.sqrt(((true - predict)**2).mean())
    
    def make_minibatch(self, N, delta, Minibatch_size, SpeedData, BrakeData=np.array([]), AccelatorData=np.array([])):
        rnd = np.random.rand(Minibatch_size)*(N-delta-1)
        source = []
        target = []
        for r in rnd:
            if BrakeData.size == 0 and AccelatorData.size == 0:
                s = [ [ SpeedData[v] ] for v in range(int(r),int(r)+delta)]
            elif AccelatorData.size == 0:
                s = [ [ SpeedData[v], BrakeData[v] ] for v in range(int(r),int(r)+delta) ]
            else:
                s = [ [ SpeedData[v], BrakeData[v], AccelatorData[v] ] for v in range(int(r),int(r)+delta) ]
            t = [ [ SpeedData[v] ] for v in range(int(r+1),int(r)+delta+1)]
            source.append(s)
            target.append(t)
        source = np.array(source)
        target = np.array(target)
        source = source.transpose(1,0,2)
        target = target.transpose(1,0,2)
        return source, target

    def Train(self):
        #parameter
        self._delta = 60
        Minibatch_size = int(self._SpeedData.size/self._delta)
        epoch = 0
        log = []

        while epoch <= self._target_epoch:
            
            if self._Input == 1:
                [source, target] = self.make_minibatch(self._SpeedData.size, self._delta, Minibatch_size, self._SpeedData)
            if self._Input == 2:
                [source, target] = self.make_minibatch(self._SpeedData.size, self._delta, Minibatch_size, self._SpeedData, BrakeData=self._BrakeData)
            if self._Input == 3:
                [source, target] = self.make_minibatch(self._SpeedData.size, self._delta, Minibatch_size, self._SpeedData, BrakeData=self._BrakeData, AccelatorData=self._AccelatorData)
            
            self._model.reset_state()
            self._model.zerograds()
            loss = 0
            for i in range(0,self._delta-1):
                s = Variable(self._xp.asarray(source[i], dtype=self._xp.float32))
                t = Variable(self._xp.asarray(target[i], dtype=self._xp.float32))
                loss += self._model(s,t)
                
            loss.backward()
            self._optimizer.update()

            if self._gpu_flag > 0:
                TrainLoss = cuda.to_cpu(loss.data)/self._delta*100
            else:
                TrainLoss = loss.data/self._delta*100
            
            self._fig.setTrainItr(epoch)
            self._fig.setTrainLoss(float(TrainLoss))
            
            self._fig.update()
            
            if epoch % self._Interrupt == 0:

                #validation
                self._model.reset_state()
                self._model.zerograds()
                validation_loss = 0
                answer = []
                    
                for i in xrange(self._TestSpeedData.size-1):
                    if self._Input == 1:
                        s = Variable( self._xp.asarray( [ [ self._TestSpeedData[i] ] ], dtype=self._xp.float32) )
                    elif self._Input == 2:
                        s = Variable( self._xp.asarray( [ [ self._TestSpeedData[i], self._TestBrakeData[i] ] ], dtype=self._xp.float32) )
                    else:
                        s = Variable( self._xp.asarray( [ [ self._TestSpeedData[i], self._TestBrakeData[i], self._TestAccelatorData[i] ] ], dtype=self._xp.float32) )
                    t = Variable( self._xp.asarray( [ [ self._TestSpeedData[i+1] ] ], dtype=self._xp.float32) )
                    ans = self._model.fwd(s)
                        
                    if self._gpu_flag >= 0:
                        loss = F.mean_squared_error(ans, t)
                        answer.append( cuda.to_cpu (ans.data[0][0] ) )
                        validation_loss += cuda.to_cpu( loss.data )
                    else:
                        answer.append( ans.data[0][0] )
                        loss = F.mean_squared_error(ans, t)
                        validation_loss += loss.data
                            
                validation_loss /= self._TestSpeedData.size*100
                
                if self._Norm:
                    answer = np.array(answer)
                else:
                    answer = np.array(answer)
                
                self._fig.setSeqPre(answer*self._Norm[0])
                self._fig.setTestLoss(float(validation_loss))
                self._fig.setTestItr(epoch)
                error = self.rmse(answer, self._TestSpeedData[1:])*100
                    
                self._fig.update()
                    
                    
                print "epoch: {}/{} now_loss : {} validatiom_loss : {} RMSE : {}".format(epoch,self._target_epoch,TrainLoss,validation_loss,error)
                log.append("epoch: {}/{} now_loss : {} validatiom_loss : {} RMSE : {}".format(epoch,self._target_epoch,TrainLoss,validation_loss,error))
                
            epoch += 1
        
        print "learning finish!"
        
        self._fig.saveFig2Image(os.path.join("./", self._Name))
        self._fig.saveFig2File(os.path.join("./", self._Name))
        
        #学習済みモデルの保存
        if self._gpu_flag >= 0:
            self._model.to_cpu()
        
        serializers.save_npz(os.path.join("./", self._Name, self._Name+".model"), self._model)
        serializers.save_npz(os.path.join("./", self._Name, self._Name+".state"), self._optimizer)    
        
        self.logoutput(os.path.join(os.path.join("./", self._Name), "TrainLog.txt"), log)
        self.Test(output=os.path.join("./", self._Name))
        self.TestSP(output=os.path.join("./", self._Name))
        

    def Test(self, output='./'):
        self._model.reset_state()
        self._model.zerograds()
        answer = []
        for i in xrange(self._TestSpeedData.size-1):
            if self._Input == 1:
                s = Variable( np.asarray( [ [ self._TestSpeedData[i] ] ], dtype=np.float32) )
            if self._Input == 2:
                s = Variable( np.asarray( [ [ self._TestSpeedData[i], self._TestBrakeData[i] ] ], dtype=np.float32) )
            if self._Input == 3:
                s = Variable( np.asarray( [ [ self._TestSpeedData[i], self._TestBrakeData[i], self._TestAccelatorData[i] ] ], dtype=np.float32) )
            ans = self._model.fwd(s)
            answer.append( ans.data[0][0] )
        answer = np.array(answer)
        
        error = self.rmse(answer, self._TestSpeedData[1:])
        print "Test RMSE : {}".format(error)
        self.logoutput(os.path.join(output, "TestLog.txt"), ["Test RMSE : {}".format(error)])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(self._TestSpeedData.size-1), self._TestSpeedData[1:]*self._Norm[0])
        ax.plot(range(self._TestSpeedData.size-1), answer*self._Norm[0])
        plt.show()
        plt.savefig(os.path.join(output, "Test.png"))
    
    #なりすましありのテスト
    def TestSP(self, sp=np.array([]), output='./'):
        if sp.size == 0:
            flag = -1
            i = 0
            s = []
            for c in range(self._TestSpeedData.size):
                if flag == 1:
                    i += 1/self._thin
                    s.append(i)
                    if i > 50:
                        flag = -1
                else:
                    i -= 1/self._thin
                    s.append(i)
                    if i < 1:    
                        flag = 1
            sp = np.array(s)
        sp = np.asarray(sp, dtype=np.float32)
        sp /= self._Norm[0]
        answer_speed = []
        answer = -1
        nextdata = [1, 10]
        
        for i in xrange(self._TestSpeedData.size-1):
            
            delta_pre = []
            
            if answer == -1:
                if self._Input == 1:
                    s = Variable( np.asarray( [ [ self._TestSpeedData[i] ] ], dtype=np.float32) )
                if self._Input == 2:
                    s = Variable( np.asarray( [ [ self._TestSpeedData[i], self._TestBrakeData[i] ] ], dtype=np.float32) )
                if self._Input == 3:
                    s = Variable( np.asarray( [ [ self._TestSpeedData[i], self._TestBrakeData[i], self._TestAccelatorData[i] ] ], dtype=np.float32) )
            else:
                if self._Input == 1:
                    s = Variable( np.asarray( [ [ answer ] ], dtype=np.float32) )
                if self._Input == 2:
                    s = Variable( np.asarray( [ [ answer, self._TestBrakeData[i] ] ], dtype=np.float32) )
                if self._Input == 3:
                    s = Variable( np.asarray( [ [ answer, self._TestBrakeData[i], self._TestAccelatorData[i] ] ], dtype=np.float32) )
                
            pre = self._model.fwd(s).data[0][0]
            nextdata[1] = self._TestSpeedData[i+1]
            
            for n in nextdata:
                delta_pre.append( self.rmse(pre, n) )
                
            arg = np.argmin(delta_pre)
            answer = nextdata[arg]
            answer_speed.append( answer )
            
            nextdata[0] = sp[i]
            
        answer_speed = np.array(answer_speed)
        error = self.rmse(answer_speed, self._TestSpeedData[1:])
        print "TestSP RMSE : {}".format(error)
        self.logoutput(os.path.join(output, "TestSPLog.txt"), ["TestSP RMSE : {}".format(error)])
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(range(self._TestSpeedData.size-1), sp[1:]*self._Norm[0], '--r')
        ax1.plot(range(self._TestSpeedData.size-1), self._TestSpeedData[1:]*self._Norm[0], '--g')
        ax1.plot(range(self._TestSpeedData.size-1), answer_speed*self._Norm[0], '-b')
        plt.show()
        plt.savefig(os.path.join(output, "TestSP.png"))


    def logoutput(self, output, l):
        with open(output, 'w') as f:
            for row in l:
                f.write(row+"\n")

if __name__ == "__main__":
    t = Trainer(True)
    t.TrainSetup("./fit_all_data.csv", "result", RNN_model1.RNN_model([100, 100], [3,1]), thin=50, target_epoch=100, Interrupt=50, Norm=[100, 100, 100])
    t.Train()
    '''
    t = Trainer(False)
    model = RNN_model1.RNN_model([100, 100], [3,1])
    serializers.load_npz("./result/result.model", model)
    t.TestSetup("./fit_all_data.csv", model, Norm=[100, 100, 100])
    t.Test()
    '''