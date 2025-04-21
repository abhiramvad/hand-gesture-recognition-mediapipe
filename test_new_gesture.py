import torch
from transformer_MAML import MAMLTransformer, TransformerClassifier

support_path = 'model/keypoint_classifier/new_gesture_support.pt'
query_path = 'model/keypoint_classifier/new_gesture_query.pt'
model_path = 'model/keypoint_classifier/maml_transformer.pt'

support_x, support_y = torch.load(support_path)
query_x, query_y = torch.load(query_path)

print("Support labels:", support_y.tolist())
print("Query labels:", query_y.tolist())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
support_x, support_y = support_x.to(device), support_y.to(device)
query_x, query_y = query_x.to(device), query_y.to(device)

# Ensure new gesture label is supported by the model (label 10)
assert support_y.max().item() <= 10, "New gesture label is out of bounds"

# Load model with 11 classes (includes new gesture)
base_model = TransformerClassifier(input_dim=2, hidden_dim=64, num_classes=11).to(device)
maml_model = MAMLTransformer(base_model, inner_lr=0.01, inner_steps=1).to(device)
maml_model.base_model.load_state_dict(torch.load(model_path))

# Run adaptation
tasks = [(support_x, support_y, query_x, query_y)]
loss, acc = maml_model(tasks)

print(f"\n🧪 Meta-learned New Gesture Evaluation")
print(f"Loss: {loss.item():.4f}")
print(f"Accuracy: {acc.item() * 100:.2f}%")
