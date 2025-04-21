import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformer_MAML import MAMLTransformer, TransformerClassifier
from sklearn.model_selection import train_test_split

# ------------------------
# Hyperparameters
# ------------------------
k_shot = 3
q_query = 5
n_way = 5  # Number of classes per task
meta_batch_size = 4
inner_lr = 0.01
meta_lr = 0.001
inner_steps = 1
epochs = 100

# ------------------------
# Load and split dataset
# ------------------------
data_path = 'model/keypoint_classifier/keypoint.csv'
data = np.loadtxt(data_path, delimiter=',', dtype='float32')
X = data[:, 1:].reshape(-1, 21, 2)
y = data[:, 0].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# ------------------------
# Build tasks from dataset
# ------------------------
def create_few_shot_task(X, y, n_way, k_shot, q_query, include_class=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    support_x, support_y, query_x, query_y = [], [], [], []

    classes = np.unique(y)
    if include_class is not None and include_class in classes:
        chosen_classes = [include_class]
        remaining = [c for c in classes if c != include_class]
        chosen_classes += list(np.random.choice(remaining, size=n_way - 1, replace=False))
    else:
        chosen_classes = np.random.choice(classes, size=n_way, replace=False)

    for cls in chosen_classes:
        idx = np.where(y == cls)[0]
        if len(idx) < k_shot + q_query:
            continue  # Skip if not enough examples
        chosen_idx = np.random.choice(idx, size=k_shot + q_query, replace=False)
        support_x.append(X_tensor[chosen_idx[:k_shot]])
        support_y.append(torch.full((k_shot,), cls))
        query_x.append(X_tensor[chosen_idx[k_shot:]])
        query_y.append(torch.full((q_query,), cls))

    support_x = torch.cat(support_x, dim=0)
    support_y = torch.cat(support_y, dim=0)
    query_x = torch.cat(query_x, dim=0)
    query_y = torch.cat(query_y, dim=0)

    return support_x, support_y, query_x, query_y


# ------------------------
# Meta Training Loop
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = TransformerClassifier(num_classes=11).to(device)
model = MAMLTransformer(base_model=base_model, inner_lr=inner_lr).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
print("Available classes in training data:", np.unique(y_train))


for epoch in range(1, epochs + 1):
    meta_loss = 0
    meta_acc = 0
    for _ in range(meta_batch_size):
        
        # 🔁 Force class 10 to be present every other epoch
        if epoch % 2 == 0:
            s_x, s_y, q_x, q_y = create_few_shot_task(X_train, y_train, n_way, k_shot, q_query, include_class=10)
        else:
            s_x, s_y, q_x, q_y = create_few_shot_task(X_train, y_train, n_way, k_shot, q_query)

        s_x, s_y, q_x, q_y = s_x.to(device), s_y.to(device), q_x.to(device), q_y.to(device)
        print(f"Classes in task: {s_y.unique().tolist()}")

        loss, acc = model.meta_update(s_x, s_y, q_x, q_y, inner_steps=inner_steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meta_loss += loss.item()
        meta_acc += acc

    print(f"[Epoch {epoch}] Meta Loss: {meta_loss / meta_batch_size:.4f}, Meta Acc: {meta_acc / meta_batch_size:.4f}")



# Save trained base model
save_path = "model/keypoint_classifier/maml_transformer.pt"
torch.save(model.base_model.state_dict(), save_path)
print(f"✅ Base model saved at: {save_path}")

