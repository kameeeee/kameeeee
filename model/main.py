# -*- coding: utf-8 -*-
import RNN_model1 #Linear->LSTM->Linearモデル
import RNN_model2 #Linear->LSTM->LSTM->Linearモデル
import Trainer
from chainer import serializers
import time



''' 
inNode = [1, 2, 3]

for i in inNode:
    start = 0
    elapsed_time = 0
    start = time.time()
    t = Trainer.Trainer(True)
    t.TrainSetup("./log/same_route/nishimura/fit_all_data.csv", "nishimura_decimation1_50node_{}".format(i), RNN_model1.RNN_model([50, 50], [i,1]), thin=1, target_epoch=1000, Interrupt=100, Norm=[100, 100, 100], TestInputFile="./log/same_route/nishimura/unknow_data.csv")
    t.Train()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]\n")


for i in inNode:
    #通常の学習(3入力)
    t = Trainer.Trainer(True)
    t.TrainSetup("./learned_data.csv", "100node_{}".format(i), RNN_model1.RNN_model([100,100], [i,1]), thin=50, target_epoch=1000, Interrupt=100, Norm=[100, 100, 100], TestInputFile="./unknow_data.csv")
    t.Train()

for i in inNode:
    #通常の学習(3入力)
    t = Trainer.Trainer(True)
    t.TrainSetup("./learned_data.csv", "10node_{}".format(i), RNN_model1.RNN_model([10,10], [i,1]), thin=50, target_epoch=1000, Interrupt=100, Norm=[100, 100, 100], TestInputFile="./unknow_data.csv")
    t.Train()
'''

t = Trainer.Trainer(False)
model = RNN_model1.RNN_model([50, 50], [3,1])
serializers.load_npz("./defore_dataset_learned/Linear_1_LSTM_1_Linear_1/50node_3/50node_3.model", model)
t.TestSetup("./log/unknow_data.csv", model, thin=1, Norm=[100, 100, 100])
TestSpeedDataSize = t._TestSpeedData.size
print TestSpeedDataSize
start1 = time.time()
t.Test()
elapsed_time1 = time.time() - start1
print ("elapsed_time:{0}".format(elapsed_time1) + "[sec]")
start2 = time.time()
t.TestSP()
elapsed_time2 = time.time() - start2
print ("elapsed_time:{0}".format(elapsed_time2) + "[sec]")
