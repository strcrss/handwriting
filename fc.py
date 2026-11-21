import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.classification import MulticlassConfusionMatrix
import time
import os
import pandas as pd
from torch.utils.data import random_split


torch.manual_seed(1)

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

train_size = int(len(train_dataset) * 0.9)
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator = torch.Generator().manual_seed(1)
)

train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True
)

train_val_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=False
)

val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle = False
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


def main():
    model = NeuralNetwork()

    os.makedirs("results", exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_time = 0

    epochs = 20

    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    history = []

    best_val_loss = float("inf")
    model_path = "results/fc_model.pth"

    for epoch in range(epochs):

        model.train()
        start_time = time.perf_counter()

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
        
        end_time = time.perf_counter()
        training_time += end_time - start_time

        model.eval()

        train_total_loss = 0
        train_correct = 0
        train_total = 0

        with torch.no_grad():
            for images,labels in train_val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                train_total_loss += loss.item() * labels.size(0)
                predicted = outputs.argmax(dim=1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

        avg_loss = train_total_loss / train_total
        accuracy = 100 * train_correct / train_total
        train_loss_history.append(avg_loss)
        train_accuracy_history.append(accuracy)


        val_total_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_total_loss += loss.item() * labels.size(0)

                predicted = outputs.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_total_loss / val_total
        val_accuracy = 100 * val_correct / val_total

        val_loss_history.append(val_avg_loss)
        val_accuracy_history.append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {avg_loss:.4f}, "
            f"Train Accuracy: {accuracy:.2f}%, "
            f"Val Loss: {val_avg_loss:4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        history.append({
            "Epoch": epoch + 1,
            "Train Loss": avg_loss,
            "Train Accuracy": accuracy,
            "Val Loss": val_avg_loss,
            "Val Accuracy": val_accuracy
        })

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), model_path)
            print("モデルを保存しました")

    print(f"学習時間 : {training_time:.2f}秒")


    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    model.eval()

    test_correct = 0
    test_total = 0

    wrong_images = []
    wrong_labels = []
    wrong_predictions = []

    metric = MulticlassConfusionMatrix(num_classes=10)

    # 勾配計算を無効にする
    with torch.no_grad():

        for images, labels in test_loader:

            # 予測
            outputs = model(images)

            # 最もスコアが高い数字を取得
            predicted = outputs.argmax(dim=1)

            # 正解数を集計
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            wrong_indices = (predicted != labels).nonzero().flatten()

            for idx in wrong_indices:
                wrong_images.append(images[idx])
                wrong_labels.append(labels[idx].item())
                wrong_predictions.append(predicted[idx].item())

            metric.update(outputs, labels)


    # 正解率を計算
    test_accuracy = 100 * test_correct / test_total

    print(f"正解率: {test_accuracy:.2f}%")

    params = sum(p.numel() for p in model.parameters())
    print("パラメータ数:", params)

    print("誤認識した枚数:", len(wrong_images))

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):

        if i < len(wrong_images):

            ax.imshow(wrong_images[i].squeeze(), cmap="gray")

            ax.set_title(f"True: {wrong_labels[i]}\n"
                f"Pred: {wrong_predictions[i]}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()

    cm = metric.compute()
    print(cm)

    plt.imshow(cm)
    plt.colorbar()

    for i in range(10):
        for j in range(10):
            plt.text(j, i, int(cm[i, j]),
                    ha="center", va="center")

    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()


    new_result = pd.DataFrame({
        "Model" : ["FC"],
        "Accuracy" : [test_accuracy],
        "Training Time" : [training_time],
        "Parameters" : [params]
    })

    csv_path = "results/comparison.csv"

    if os.path.exists(csv_path):
        old_results = pd.read_csv(csv_path)
        results = pd.concat([old_results, new_result], ignore_index=True)
    else:
        results = new_result

    results.to_csv(csv_path, index=False)

    plt.plot(range(1, epochs+1), train_loss_history, marker = "o", label = "Train")
    plt.plot(range(1, epochs+1), val_loss_history, marker = "o", label = "Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FC: Training and Validation Loss")
    plt.legend()
    plt.savefig("results/fc_loss.png")
    plt.show()

    plt.plot(range(1, epochs+1), train_accuracy_history, marker = "o", label = "Train")
    plt.plot(range(1, epochs+1), val_accuracy_history, marker = "o", label = "Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("FC: Training and Validation Accuracy")
    plt.legend()
    plt.savefig("results/fc_accuracy.png")
    plt.show()

    history_df = pd.DataFrame(history)

    history_df.to_csv("results/fc_history.csv", index=False)


if __name__ == "__main__":
    main()