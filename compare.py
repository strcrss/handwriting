import pandas as pd
import matplotlib.pyplot as plt


fc_df = pd.read_csv("results/fc_history.csv")
cnn_df = pd.read_csv("results/cnn_history.csv")


plt.plot(fc_df["Epoch"], fc_df["Train Loss"], marker="o", label="FC Train")
plt.plot(fc_df["Epoch"], fc_df["Val Loss"], marker="o", label="FC Validation")

plt.plot(fc_df["Epoch"], cnn_df["Train Loss"], marker="o", label="CNN Train")
plt.plot(fc_df["Epoch"], cnn_df["Val Loss"], marker="o", label="CNN Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("FC vs CNN: Loss")
plt.legend()
plt.savefig("results/loss_comparison.png")
plt.show()


plt.plot(fc_df["Epoch"], fc_df["Train Accuracy"], marker="o", label="FC Train")
plt.plot(fc_df["Epoch"], fc_df["Val Accuracy"], marker="o", label="FC Validation")

plt.plot(fc_df["Epoch"], cnn_df["Train Accuracy"], marker="o", label="CNN Train")
plt.plot(fc_df["Epoch"], cnn_df["Val Accuracy"], marker="o", label="CNN Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("FC vs CNN: Accuracy")
plt.legend()
plt.savefig("results/accuracy_comparison.png")
plt.show()
