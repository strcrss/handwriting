import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# MNISTの読み込み
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.view(-1, 784)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


model = NeuralNetwork()

# 訓練データから画像を1枚取り出す
image, label = train_dataset[0]

# モデルに画像を入力
model.eval()

with torch.no_grad():
    output = model(image)

# 結果を表示
print("入力画像の形状:", image.shape)
print("モデルの出力形状:", output.shape)
print("正解ラベル:", label)
print("モデルの出力:", output)

# 最も大きい値を持つクラスを予測結果とする
prediction = output.argmax(dim=1).item()

print("予測した数字:", prediction)

# 出力を確率に変換
probabilities = torch.softmax(output, dim=1)

print("各数字の確率:")
for i, probability in enumerate(probabilities[0]):
    print(f"{i}: {probability.item():.4f}")

# 入力画像を表示
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Correct: {label}, Predicted: {prediction}")
plt.axis("off")
plt.show()