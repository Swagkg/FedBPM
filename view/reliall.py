import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import random

rcParams['font.family']='Times New Roman'
train_path = "../leaf/data/femnist/data/test"
files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.json')]


all_data = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        all_data.append(data)

client_ids = []
all_labels = set()

for data in all_data:
    for user in data['users']:
        client_ids.append(user)
        all_labels.update(data['user_data'][user]['y'])

# selected_clients = random.sample(client_ids,10)
all_labels = sorted(list(all_labels))


label_distribution = np.zeros((len(client_ids), len(all_labels)))

client_idx = 0
for data in all_data:
    for user in data['users']:
        y_data = data['user_data'][user]['y']
        unique_labels, counts = np.unique(y_data, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_idx = all_labels.index(label)
            label_distribution[client_idx, label_idx] = count
        client_idx += 1
 # normalized_distribution = distribution_matrix / (distribution_matrix.sum(axis=1, keepdims=True) + 1e-8)

fig_width = max(12, len(all_labels) * 0.3)
fig_height = max(8, len(client_ids) * 0.02)

plt.figure(figsize=(12, 10))
sns.heatmap(
    label_distribution,
    cmap="YlGnBu",
    xticklabels=all_labels,
    yticklabels=client_ids,
    cbar_kws={'label': 'Number of Samples'},
    linewidths=0.5,
    linecolor='gray',
    annot=True,
    fmt="g",
    annot_kws={"size":9}

)

plt.title("Label Distribution Across Clients", fontsize=16)
plt.xlabel("Labels", fontsize=12)
plt.ylabel("Test Clients", fontsize=12)
plt.tight_layout()

plt.show()