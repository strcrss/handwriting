# Handwriting Recognition

PytorchのデータセットMNISTを使用し、ニューラルネットワークによる画像分類を実装しています。

## 使用技術
Python  
Pytorch  
Torchvision  
Matplotlib  
TorchMetrics  

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
上記の設定で学習を行った結果、97.61%の分類精度を達成
|評価項目|FC|CNN|
|---|---:|---:|
|正解率|97.21%|99.09%|
|誤認識数|279|91|
|パラメータ数|101770|421642|

混同行列による分類傾向の分析と、誤認識画像の可視化を実装しました。  

## 今後の予定
CNNの実装と性能比較

