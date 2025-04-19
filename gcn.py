import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10

# Define hand graph edges for 21 landmarks

def get_edge_index():
    edges = [
        (0,1),(1,2),(2,3),(3,4),        # Thumb
        (0,5),(5,6),(6,7),(7,8),        # Index
        (0,9),(9,10),(10,11),(11,12),    # Middle
        (0,13),(13,14),(14,15),(15,16),  # Ring
        (0,17),(17,18),(18,19),(19,20),  # Pinky
        (5,9),(9,13),(13,17)             # Palm
    ]
    # Duplicate edges for undirected graph
    edge_index = torch.tensor(edges + [(j,i) for i,j in edges], dtype=torch.long).t().contiguous()
    return edge_index

class GCNClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    labels = df.iloc[:, 0].astype(int).values
    features = df.iloc[:, 1:].astype(float).values.astype(np.float32)
    # Map original labels to 0...num_classes-1
    unique_labels = sorted(np.unique(labels))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    data_list = []
    edge_index = get_edge_index()
    for feat, label in zip(features, labels):
        x = torch.tensor(feat.reshape(21, 2), dtype=torch.float)
        y = torch.tensor(label_map[label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list, len(unique_labels)


def train_model(data_list, num_classes):
    labels = [d.y.item() for d in data_list]
    train_idx, test_idx = train_test_split(
        list(range(len(data_list))), test_size=0.2,
        stratify=labels, random_state=RANDOM_SEED
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1,
        stratify=[labels[i] for i in train_idx], random_state=RANDOM_SEED
    )
    train_ds = [data_list[i] for i in train_idx]
    val_ds = [data_list[i] for i in val_idx]
    test_ds = [data_list[i] for i in test_idx]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = GCNClassifier(input_dim=2, hidden_dim=64, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, MAX_EPOCHS + 1):
        # Training
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.y.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
        val_loss /= total
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'model/keypoint_classifier/gcn.pt')
            print(f"Model saved with Val Acc: {val_acc:.4f}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping after {epoch} epochs")
                break

    # Plotting
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.legend()
    os.makedirs('model/keypoint_classifier/plots', exist_ok=True)
    plt.savefig('model/keypoint_classifier/plots/gcn_loss.png')

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig('model/keypoint_classifier/plots/gcn_acc.png')

    # Testing
    model.load_state_dict(torch.load('model/keypoint_classifier/gcn.pt'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1).tolist()
            labels = batch.y.tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('model/keypoint_classifier/plots/gcn_cm.png')

if __name__ == '__main__':
    data_list, num_classes = load_dataset('gesture_data.csv')
    print(f"Detected {num_classes} classes")
    train_model(data_list, num_classes)