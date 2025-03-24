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
model_save_path = 'model/keypoint_classifier/transformer.pt'
output_dir = 'model/keypoint_classifier/plots'
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Load dataset
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, 43)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# Determine the number of classes
NUM_CLASSES = len(np.unique(y_dataset))
print(f"Detected {NUM_CLASSES} unique classes in the dataset: {np.unique(y_dataset)}")

# Compute class weights to handle imbalance
class_counts = np.bincount(y_dataset)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 21, 2)  # [samples, 21 keypoints, 2 coords]
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 21, 2)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=NUM_CLASSES, num_heads=4, num_layers=2, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        
        # Project input keypoints to higher dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len=21, input_dim=2]
        h = self.input_proj(x)  # [batch_size, 21, hidden_dim]
        h = F.relu(h)
        
        # Transformer encoding
        h = self.transformer_encoder(h)  # [batch_size, 21, hidden_dim]
        
        # Pooling: take mean across sequence dimension
        h = h.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)  # [batch_size, num_classes]
        return h

# Initialize model, optimizer, and loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerClassifier(input_dim=2, hidden_dim=64, num_classes=NUM_CLASSES, num_heads=4, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training metrics
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Early stopping parameters
patience = 20
best_val_acc = 0.0
patience_counter = 0

# Training loop
def train():
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
    return total_loss / len(train_loader)

# Evaluation
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    y_true, y_pred = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss, y_true, y_pred

# Training and evaluation loop
for epoch in range(1000):
    train_loss = train()
    train_acc, train_loss_eval, _, _ = evaluate(train_loader)
    val_acc, val_loss, y_true, y_pred = evaluate(test_loader)
    
    # Store metrics
    train_losses.append(train_loss_eval)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved with Val Acc: {best_val_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Final evaluation
val_acc, _, y_true, y_pred = evaluate(test_loader)
print(f"Final Test Accuracy: {val_acc:.4f}")

# Plotting and saving training metrics
def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

# Confusion Matrix
def print_confusion_matrix(y_true, y_pred, output_dir):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    plt.ylim(len(labels), 0)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save classification report to text file
    report = classification_report(y_true, y_pred)
    print('Classification Report')
    print(report)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

# Generate and save plots
plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir)
print_confusion_matrix(y_true, y_pred, output_dir)