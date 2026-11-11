# Handwriting Recognition

PytorchのデータセットMNISTを使用し、ニューラルネットワークによる画像分類を実装しています。

## 使用技術
Python
Pytorch
Torchvision
Matplotlib

## モデル構成
入力層 : 784(28 × 28)
全結合層 : 128
活性化関数 : ReLU
出力層 : 10

損失関数 : Cross Entropy Loss
最適化手法 : Adam
学習率 : 0.001
Epoch数 : 5
Batch size : 64

## 評価結果
上記の設定で学習を行った結果、97.61%の分類精度を達成

## 今後の予定
CNNの実装と性能比較
