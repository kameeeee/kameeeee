# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

class TrainFigure:
    def __init__(self, flag, SeqData):        
        #学習
        self._TrainLoss = []
        self._Train_x = []
        
        #検証
        self._TestLoss = []
        self._Test_x = []
        
        #シーケンステスト
        self._Seq_x = SeqData[0]
        self._Seq_y = SeqData[1]
        self._Seq_pre = np.zeros(np.size(SeqData[0]))
        
        self._init = False
        
        if flag:
            self._fig = plt.figure()
            self._ax1 = self._fig.add_subplot(211)
            plt.title('Train(Separated) and Test(Sequence) Loss', fontsize=18)
            plt.xlabel("Epoch", fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            self._ax2 = self._fig.add_subplot(212)
            plt.title("Test at Squence Data", fontsize=18)
            plt.xlabel("Time", fontsize=16)
            plt.ylabel("Data", fontsize=16)
            
            self._ax1.set_ylim((0,np.max(self._Seq_y)*1.2))
            
            plt.tight_layout()
            
            plt.show()
    
    def setTrainItr(self, Itr):
        self._Train_x.append(Itr)
        
    def setTrainLoss(self, Loss):
        self._TrainLoss.append(Loss)
        
    def setTestItr(self, Itr):
        self._Test_x.append(Itr)
        
    def setTestLoss(self, Loss):
        self._TestLoss.append(Loss)
    
    def setSeqPre(self, pre):
        self._Seq_pre = pre
    
    def saveFig2Image(self, fpath):
        plt.savefig(os.path.join(fpath, "Loss.png"))
    
    def saveFig2File(self, fpath):
        with open(os.path.join(fpath,"Train.csv"), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for row in zip(self._Train_x, self._TrainLoss):
                writer.writerow(row)
        
        with open(os.path.join(fpath,"Test.csv"), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for row in zip(self._Test_x, self._TestLoss):
                writer.writerow(row)
    
    def update(self):
        if self._init:
            self._lines1.set_data(self._Train_x, self._TrainLoss)
            self._lines2.set_data(self._Test_x, self._TestLoss)
            self._lines4.set_data(self._Seq_x, self._Seq_pre)
            self._ax1.set_xlim((0,np.max(self._Train_x)))
            self._ax1.set_ylim((0,np.max(self._TrainLoss)))
            plt.pause(.001)
        else:
            self._lines1, = self._ax1.plot(self._Train_x, self._TrainLoss, '-b', label="Train")
            self._lines2, = self._ax1.plot(self._Test_x, self._TestLoss, '-r', label="Test")
            self._lines3, = self._ax2.plot(self._Seq_x, self._Seq_y, '-b', label="Sequence Input")
            self._lines4, = self._ax2.plot(self._Seq_x, self._Seq_pre, '-r', label="Predict")
            self._ax1.legend(loc="upper left")
            self._ax2.legend(loc="upper left")
            self._init = True