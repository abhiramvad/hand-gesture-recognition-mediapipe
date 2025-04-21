import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os

# Paths
csv_path = 'model/keypoint_classifier/keypoint.csv'  # adjust if needed
support_out = 'model/keypoint_classifier/new_gesture_support.pt'
query_out = 'model/keypoint_classifier/new_gesture_query.pt'

# Load the CSV
data = np.loadtxt(csv_path, delimiter=',', dtype='float32')

# Filter only label 10
label = 10
gesture_data = data[data[:, 0] == label]

print(f"[INFO] Total samples found for label {label}: {len(gesture_data)}")
assert len(gesture_data) >= 10, "You need at least 10 samples for splitting!"

# Extract features and labels
X = gesture_data[:, 1:].reshape(-1, 21, 2)
y = gesture_data[:, 0].astype(int)

# Split into 5 support, 10 query (you can change train_size as needed)
X_support, X_query, y_support, y_query = train_test_split(X, y, train_size=5, random_state=42)

# Convert to tensors
support_x = torch.tensor(X_support, dtype=torch.float32)
support_y = torch.tensor(y_support, dtype=torch.long)
query_x = torch.tensor(X_query, dtype=torch.float32)
query_y = torch.tensor(y_query, dtype=torch.long)

# Save .pt files
os.makedirs(os.path.dirname(support_out), exist_ok=True)
torch.save((support_x, support_y), support_out)
torch.save((query_x, query_y), query_out)

print(f"✅ Saved support set to: {support_out}")
print(f"✅ Saved query set to: {query_out}")