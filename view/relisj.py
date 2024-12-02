import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random

rcParams['font.family']='Times New Roman'
train_path = "../leaf/data/femnist/data/test"


files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.json')]

all_data = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        all_data.append(data)

client_data = {}
for data in all_data:
    for user in data['users']:
        y_data = data['user_data'][user]['y']
        unique_labels, counts = np.unique(y_data, return_counts=True)
        client_data[user] = dict(zip(unique_labels, counts))

selected_clients = random.sample(list(client_data.keys()), 10)

all_labels = set()
for user in client_data:
    all_labels.update(client_data[user].keys())

selected_labels = random.sample(sorted(all_labels), 20)

distribution_matrix = np.zeros((len(selected_clients), len(selected_labels)))
for i, client in enumerate(selected_clients):
    for j, label in enumerate(selected_labels):
        distribution_matrix[i, j] = client_data[client].get(label, 0)

    normalized_distribution = distribution_matrix / (distribution_matrix.sum(axis=1, keepdims=True) + 1e-8)

plt.figure(figsize=(12, 6))
sns.heatmap(
    normalized_distribution,
    cmap="YlGnBu",
    xticklabels=selected_labels,
    yticklabels=selected_clients,
    cbar_kws={'label': 'Normalized Proportion'},
    linewidths=0.2,
    linecolor='gray',
    annot=True,
    fmt=".2f",
    annot_kws={"size":8}
)


plt.title("Label Distribution Heatmap", fontsize=16)
plt.xlabel("Labels", fontsize=12)
plt.ylabel("Test Clients", fontsize=12)
plt.tight_layout()


plt.show()