# -*- coding: utf-8 -*-
import RNN_model1 #Linear->LSTM->Linearモデル
import RNN_model2 #Linear->LSTM->LSTM->Linearモデル
import Trainer
from chainer import serializers

'''
-----定義されたクラスの初期化引数-----

RNN_model
・層間結合の本数    ：   List
    [1層目の出力(=2層目の入力), 2層目の以下略, 3層目の・・・]
・入力と出力  ：   List
    [入力, 出力]
    

-----クラスの使い方-----
クラスの初期化
t = Trainer.Trainer(train)
    ・train : 学習する場合はTrue, しないときはFalse

学習セットアップ
t.TrainSetup(inputFileName, Name, model, thin=50, Norm=[1, 1, 1], delta=60, target_epoch=500, Interrupt=100)
    ・inputFileName(1番目の引数)  :   String
        入力データのCSVファイル名
    ・Name(2番目の引数)   ：   String
        出力先フォルダ名
    ・model(3番目の引数)  ：   ModelClass Object
        モデルオブジェクトを渡す
    ・thin=50    ：   int
        間引きの数(初期値50)
    ・Norm=[1, 1, 1]  ：   List
        正規化するときの割る数(スピード、ブレーキ、アクセル)
    ・delta=60   :   int
        学習時のデルタ(初期値60)
    ・target_epoch=500   ：   int
        学習エポック数(初期値500)
    ・Interrupt=100  ：int
        Validationを入れるタイミング
    ・TestInputFile=""   ：   string
        テスト入力データのCSVファイル名

学習開始
t.Train()

テストのみのときはこちらを呼び出し
t.TestSetup(inputFileName, model, thin=50, Norm=[1, 1, 1])
    同じ名前の引数は上を参照
    第二引数のmodelはロード済みのものをわたすべし(must)

以下２つは必ずTestSetupを呼んでから呼ぶこと(テストのみのとき)
t.Test(output='./')
    ・output ：   string
        出力グラフの保存先フォルダを渡す

t.TestSP(sp=np.array([]), output='./')
    ・sp ：   numpy.ndarray
        テストデータと同じ長さの攻撃データを渡す
'''


#通常の学習(3入力)
t = Trainer.Trainer(True)
t.TrainSetup("./fit_all_data.csv", "result", RNN_model1.RNN_model([10, 10], [3,1]), thin=50, target_epoch=500, Interrupt=100, Norm=[100, 100, 100])
t.Train()

'''
#LSTM2層モデルの学習(1入力)
t = Trainer.Trainer(True)
t.TrainSetup("./fit_all_data.csv", "result2", RNN_model2.RNN_model([100, 100, 100], [1,1]), thin=50, target_epoch=500, Interrupt=100)
t.Train()

#データのテストのみするとき
t = Trainer.Trainer(False)
model = RNN_model1.RNN_model([100, 100], [3,1])
serializers.load_npz("./result/result.model", model)
t.TestSetup("./fit_all_data.csv", model, Norm=[100, 100, 100])
TestSpeedDataSize = t._TestSpeedData.size
print TestSpeedDataSize
t.Test()
t.TestSP()
'''