import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths
dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/graph_transformer.pt'
output_dir = 'model/keypoint_classifier/plots'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, 43)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0,))

# Determine the number of classes
NUM_CLASSES = len(np.unique(y_dataset))
print(f"Detected {NUM_CLASSES} unique classes in the dataset: {np.unique(y_dataset)}")

# Compute class weights
class_counts = np.bincount(y_dataset)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 21, 2)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 21, 2)
y_test = torch.tensor(y_test, dtype=torch.long)

# Show class frequency split
train_class_counts = np.bincount(y_train.numpy())
test_class_counts = np.bincount(y_test.numpy())
print("\n📊 Class Frequency (Train Set):")
for i, count in enumerate(train_class_counts):
    print(f"Class {i}: {count} samples")
print("\n📊 Class Frequency (Test Set):")
for i, count in enumerate(test_class_counts):
    print(f"Class {i}: {count} samples")

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Graph Transformer Model
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x_res = x
        x, _ = self.attention(x, x, x)
        x = self.norm1(x_res + self.dropout(x))
        x_res = x
        x = self.ffn(x)
        x = self.norm2(x_res + self.dropout(x))
        return x

class GraphTransformerClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=NUM_CLASSES, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(hidden_dim * 21, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.adj = self.create_hand_adjacency_matrix()

    def create_hand_adjacency_matrix(self):
        adj = torch.zeros(21, 21)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        for i, j in connections:
            adj[i, j] = 1
            adj[j, i] = 1
        adj += torch.eye(21)
        return adj

    def forward(self, x):
        adj = self.adj.to(x.device)
        x = self.relu(self.input_proj(x))
        for layer in self.layers:
            x = layer(x, adj)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphTransformerClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc, patience_counter, patience = 0.0, 0, 20

# Train and evaluate
for epoch in range(1000):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Eval
    model.eval()
    correct, total, y_true, y_pred, val_loss = 0, 0, [], [], 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_acc = correct / total
    train_acc = (model(X_train.to(device)).argmax(1) == y_train.to(device)).float().mean().item()
    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss / len(test_loader))
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), model_save_path)
        print("✅ Model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping.")
            break

# Final eval
model.load_state_dict(torch.load(model_save_path))
model.eval()
val_acc = (model(X_test.to(device)).argmax(1) == y_test.to(device)).float().mean().item()
y_pred_final = model(X_test.to(device)).argmax(1).cpu().numpy()
y_true_final = y_test.cpu().numpy()

# Plot
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# Classification report as text
report = classification_report(y_true_final, y_pred_final, digits=2, output_dict=False)
axs[0, 0].axis('off')
axs[0, 0].text(0.01, 1.0, str(report), {'fontsize': 11}, fontproperties='monospace')
axs[0, 0].set_title("Classification Report", fontsize=13)

# Accuracy
axs[0, 1].plot(train_accs, label="Train Accuracy")
axs[0, 1].plot(val_accs, label="Validation Accuracy")
axs[0, 1].set_title("Training and Validation Accuracy")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Accuracy")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Loss
axs[1, 0].plot(train_losses, label="Train Loss")
axs[1, 0].plot(val_losses, label="Validation Loss")
axs[1, 0].set_title("Training and Validation Loss")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Confusion matrix
cm = confusion_matrix(y_true_final, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', ax=axs[1, 1], cmap='Blues')
axs[1, 1].set_title("Confusion Matrix")
axs[1, 1].set_xlabel("Predicted")
axs[1, 1].set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_report_graph_transformer.png'))
plt.show()
print("📊 Graph Transformer report saved to full_report_graph_transformer.png")
