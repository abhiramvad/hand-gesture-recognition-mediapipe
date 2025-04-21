import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Settings
data_path = 'model/keypoint_classifier/keypoint.csv'
output_path = 'model/keypoint_classifier/test_tasks.pt'
shots = 5  # number of examples per class in support set
queries = 15  # number of examples per class in query set

assert os.path.exists(data_path), f"{data_path} not found"

# Load dataset
X = np.loadtxt(data_path, delimiter=',', dtype='float32', usecols=range(1, 43))
y = np.loadtxt(data_path, delimiter=',', dtype='int32', usecols=(0,))
X = torch.tensor(X, dtype=torch.float32).view(-1, 21, 2)
y = torch.tensor(y, dtype=torch.long)

unique_classes = torch.unique(y)
meta_tasks = []

for cls in unique_classes:
    indices = (y == cls).nonzero(as_tuple=True)[0]
    if len(indices) < shots + queries:
        print(f"Skipping class {cls.item()} due to insufficient samples")
        continue

    cls_indices = indices[torch.randperm(len(indices))]
    support_idx = cls_indices[:shots]
    query_idx = cls_indices[shots:shots+queries]

    support_x, support_y = X[support_idx], y[support_idx]
    query_x, query_y = X[query_idx], y[query_idx]

    meta_tasks.append((support_x, support_y, query_x, query_y))

torch.save(meta_tasks, output_path)
print(f"✅ Saved {len(meta_tasks)} meta-evaluation tasks to: {output_path}")
