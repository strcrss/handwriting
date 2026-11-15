import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
import time
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split


torch.manual_seed(1)

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

train_subnet, val_subnet = random_split(
    train_dataset,
    [train_size, val_size],
    generator = torch.Generator().manual_seed(1)
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

val_loader = DataLoader(
    val_subnet,
    batch_size=64,
    shuffle = True
)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

model = CNN()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start_time = time.perf_counter()

epochs = 5

train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []

for epoch in range(epochs):

    model.train()

    train_total_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    avg_loss =  train_total_loss / len(train_loader)
    accuracy = 100 * train_correct / train_total
    train_accuracy_history.append(accuracy)
    train_loss_history.append(avg_loss)

    model.eval()

    val_total_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_avg_loss = val_total_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    val_loss_history.append(val_avg_loss)
    val_accuracy_history.append(val_accuracy)


    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {avg_loss:.4f}"
        f"Train Accuracy: {accuracy:.2f}%,"
        f"Val Loss: {val_avg_loss:4f},"
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

end_time = time.perf_counter()
training_time = end_time - start_time

print(f"学習時間 : {training_time:.2f}秒")

model.eval()

test_correct = 0
test_total = 0

wrong_images = []
wrong_labels = []
wrong_predictions = []

metric = MulticlassConfusionMatrix(num_classes=10)

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predicted = outputs.argmax(dim=1)

        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        wrong_indices = (predicted != labels).nonzero().flatten()

        for idx in wrong_indices:
            wrong_images.append(images[idx])
            wrong_labels.append(labels[idx].item())
            wrong_predictions.append(predicted[idx].item())

        metric.update(outputs, labels)

test_accuracy = 100 * test_correct / test_total

print(f"正解率: {test_accuracy:2f}%")

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


os.makedirs("results", exist_ok=True)

new_result = pd.DataFrame({
    "Model" : ["CNN"],
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


plt.plot(range(1, epochs+1), train_loss_history, marker = "o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("results/cnn_loss.png")
plt.show()

plt.plot(range(1, epochs+1), train_loss_history, marker = "o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.savefig("results/cnn_accuracy.png")
plt.show()