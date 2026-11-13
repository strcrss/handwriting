import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/comparison.csv")

summary = df.groupby("Model").agg({
    "Accuracy" : ["mean", "std"],
    "Training Time" : ["mean", "std"],
    "Parameters" : "mean"
})

print(summary)

summary.to_csv("results/summary.csv")

accuracy_mean = summary["Accuracy"]["mean"]
accuracy_std = summary["Accuracy"]["std"]

plt.figure()
plt.bar(accuracy_mean.index, accuracy_mean.values, yerr = accuracy_std.values)
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(90, 100)
plt.savefig("results/accuracy.png")
plt.show()

time_mean = summary["Training Time"]["mean"]
time_std = summary["Training Time"]["std"]

plt.figure()
plt.bar(time_mean.index, time_mean.values, yerr = time_std.values)
plt.xlabel("Model")
plt.title("Training Time Comparison")
plt.ylabel("Training Time (s)")
plt.savefig("results/training_time.png")
plt.show()


