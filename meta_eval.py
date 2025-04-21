import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformer_MAML import MAMLTransformer, TransformerClassifier  # Correct imports

# Settings
MODEL_PATH = 'model/keypoint_classifier/maml_transformer.pt'
TEST_DATA_PATH = 'model/keypoint_classifier/test_tasks.pt'
PLOT_DIR = 'model/keypoint_classifier/meta_eval_plots'

os.makedirs(PLOT_DIR, exist_ok=True)

# Load data (each task is (X_train, y_train, X_test, y_test))
test_tasks = torch.load(TEST_DATA_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base model and MAML wrapper
base_model = TransformerClassifier(input_dim=2, hidden_dim=64, num_classes=10)
base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
maml = MAMLTransformer(base_model).to(device)
maml.eval()

# Collect predictions
y_true_all = []
y_pred_all = []

print(f"[INFO] Evaluating on {len(test_tasks)} tasks...")

for X_train, y_train, X_test, y_test in test_tasks:
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    fast_weights = list(base_model.parameters())
    for _ in range(1):  # 1 inner loop step
        outputs = base_model(X_train)
        loss = nn.functional.cross_entropy(outputs, y_train)
        grads = torch.autograd.grad(loss, base_model.parameters(), create_graph=False)
        fast_weights = [w - 0.01 * g for w, g in zip(fast_weights, grads)]

    # Apply fast weights manually
    def forward_with_weights(x, weights):
        h = nn.functional.linear(x, weights[0], weights[1])
        h = nn.functional.relu(h)
        h = base_model.transformer_encoder(h)
        h = h.mean(dim=1)
        h = nn.functional.linear(nn.functional.dropout(
            nn.functional.relu(nn.functional.linear(h, weights[4], weights[5])), p=0.2),
            weights[6], weights[7])
        return h

    with torch.no_grad():
        preds = forward_with_weights(X_test, fast_weights).argmax(dim=1)

    y_true_all.extend(y_test.cpu().numpy())
    y_pred_all.extend(preds.cpu().numpy())

# Report
print("\n📋 Meta-Evaluation Classification Report:")
report = classification_report(y_true_all, y_pred_all)
print(report)

with open(os.path.join(PLOT_DIR, 'meta_classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Meta-Evaluation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'meta_confusion_matrix.png'))
plt.close()

print(f"\n✅ Confusion matrix and report saved to: {PLOT_DIR}")
