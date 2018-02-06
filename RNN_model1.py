# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L

class RNN_model(chainer.Chain):
    def __init__(self, param, io):
        self._io = io
        super(RNN_model, self).__init__(
            l1 = L.Linear(io[0],param[0]),
            l2 = L.LSTM(param[0],param[1]), #LSTM
            l3 = L.Linear(param[1],io[1]),
        )
    
    #回帰問題なので二乗平均誤差
    def __call__(self, x, t):
        return F.mean_squared_error(self.fwd(x), t)
    
    #順伝播の計算
    def fwd(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return h3
        
    #LSTMの状態をリセット
    def reset_state(self):
        self.l2.reset_state()
    
    def InputNum(self):
        return self._io[0]