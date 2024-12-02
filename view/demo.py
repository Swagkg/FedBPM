import os
import json
import numpy as np
import matplotlib.pyplot as plt

train_path = "../leaf/data/femnist/data/train"

files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.json')]

all_data = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        all_data.append(data)

cont = 0
for data in all_data:
    for user in data['users']:
        print(f"Client {user} has {len(data['user_data'][user]['y'])} samples")
        cont += 1
        y_data = data['user_data'][user]['y']  # 客户端的标签数据
        unique_labels, counts = np.unique(y_data, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))

        print(f"Client {user} label distribution: {label_dist}")

print(f"Total number of clients: {cont}")

# for data in all_data:
#     users = data['users']  # 客户端名称
#     user_data = data['user_data']  # 客户端数据
#     num_samples = data['num_samples']  # 每个客户端的数据数量
#
#     print(f"Number of users: {len(users)}")
#     for user in users:
#         x_data = user_data[user]['x']  # 特征
#         y_data = user_data[user]['y']  # 标签
#
#         # 打印每个客户端的信息
#         print(f"Client {user}:")
#         print(f"  Number of samples: {len(y_data)}")
#         print(f"  Unique labels: {set(y_data)}")



