import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim

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

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

criterion = nn.CrossEntropyLoss()

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

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習回数
epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for images, labels in train_loader:

        # 予測
        outputs = model(images)

        # 損失計算
        loss = criterion(outputs, labels)

        # 勾配の初期化
        optimizer.zero_grad()

        # 逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()

        total_loss += loss.item()

    # 1 epochあたりの平均損失
    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {avg_loss:.4f}"
    )

    # モデルを評価モードにする
model.eval()

correct = 0
total = 0

wrong_images = []
wrong_labels = []
wrong_predictions = []

# 勾配計算を無効にする
with torch.no_grad():

    for images, labels in test_loader:

        # 予測
        outputs = model(images)

        # 最もスコアが高い数字を取得
        predicted = outputs.argmax(dim=1)

        # 正解数を集計
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 間違えた画像の位置を取得
        wrong_indices = (predicted != labels).nonzero().flatten()

        for idx in wrong_indices:
            wrong_images.append(images[idx])
            wrong_labels.append(labels[idx].item())
            wrong_predictions.append(predicted[idx].item())

# 正解率を計算
accuracy = 100 * correct / total

print(f"正解率: {accuracy:.2f}%")

print("誤認識した枚数:", len(wrong_images))

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):

    ax.imshow(wrong_images[i].squeeze(), cmap="gray")

    ax.set_title(
        f"True: {wrong_labels[i]}\n"
        f"Pred: {wrong_predictions[i]}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()