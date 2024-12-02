import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import random

rcParams['font.family']='Times New Roman'
train_path = "../leaf/data/cifar10/data/test"


files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.json')]

all_data = []
for file in files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    except Exception as e:
        print(f"Error loading {file}: {e}")

cont = 0
all_clients_label_dist = {}

for data in all_data:
    for user in data['users']:
        cont += 1

        y_data = data['user_data'][user]['y']
        unique_labels, counts = np.unique(y_data, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))

        all_clients_label_dist[user] = label_dist
select_clients = random.sample(list(all_clients_label_dist.keys()),50)

all_labels = sorted({label for client_labels in all_clients_label_dist.values() for label in client_labels})

# client_names = list(all_clients_label_dist.keys())
stacked_data = []
for label in all_labels:
    label_counts = [all_clients_label_dist[client].get(label, 0) for client in select_clients]
    stacked_data.append(label_counts)


plt.figure(figsize=(12, 6))
x = np.arange(len(select_clients))
bottom = np.zeros(len(select_clients))
bar_width = 0.5

# color_map = cm.get_cmap("tab20", len(all_labels))
# colors = [color_map(i)[:3] for i in range(len(all_labels))]

for i, label_counts in enumerate(stacked_data):
    plt.bar(x, label_counts, width = bar_width, bottom=bottom, label=f'Label {all_labels[i]}', alpha=1)
    bottom += np.array(label_counts)


plt.xlabel('Test Clients')
plt.ylabel('Sample Count')
plt.title('Stacked Bar Chart of Label Distribution Across Clients')
plt.xticks(ticks=x, labels=select_clients, rotation=45)
plt.legend(title="Labels", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title_fontsize=10)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=1)

# plt.savefig("label_distribution.eps", format='eps', bbox_inches='tight')
# print("Plot saved as 'label_distribution.eps'")

plt.show()