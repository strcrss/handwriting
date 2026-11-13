# Handwriting Recognition

PytorchのデータセットMNISTを使用し、ニューラルネットワークによる画像分類を実装しています。  
全結合ニューラルネットワーク(FC)と畳み込みニューラルネットワーク(CNN)の二種類を作成し、その認識精度・学習時間とパラメータ数を比較しました。

## 使用技術
Python  
Pytorch  
Torchvision  
Matplotlib  
TorchMetrics  
pandas  

## モデル構成

### FC
入力層 : 784(28 × 28)  
全結合層 : (784, 128)  
活性化関数 : ReLU  
全結合層 : (128, 10)  
出力層 : 10  
  
### CNN  
畳み込み層 : 32  
活性化関数 : ReLU  
Max Pooling : 2  
畳み込み層 : 64  
活性化関数 : ReLU  
Max Pooling : 2  
平坦化層 : Flatten  
全結合層 : (3136, 128)  
活性化関数 : ReLU  
全結合層 : (128, 10)

## 学習条件
損失関数 : Cross Entropy Loss  
最適化手法 : Adam  
学習率 : 0.001  
Epoch数 : 5  
Batch size : 64  

## 評価結果
|評価項目|FC|CNN|
|---|---:|---:|
|正解率|97.22%|98.96%|
|誤認識数|278|104|
|パラメータ数|101770|421642|

混同行列による分類傾向の分析と、誤認識画像の可視化を実装しました。  

## 今後の予定
CNNの実装と性能比較

