#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.func import functional_call
from maml_graph_transformer import MAMLGraphTransformer
from graph_transformer import GraphTransformerClassifier

# 1) Load your support & query sets (pre‑saved .pt files)
support_x, support_y = torch.load('model/keypoint_classifier/new_gesture_support.pt')
query_x,   query_y   = torch.load('model/keypoint_classifier/new_gesture_query.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
support_x, support_y = support_x.to(device), support_y.to(device)
query_x,   query_y   = query_x.to(device),   query_y.to(device)

# 2) Instantiate the base G‑Transformer & load the MAML‑trained weights
num_classes = 11
base_model = GraphTransformerClassifier(input_dim=2, hidden_dim=64, num_classes=num_classes)
base_model.load_state_dict(torch.load('model/keypoint_classifier/maml_graph_transformer.pt',
                                      map_location=device))
base_model.to(device).eval()

# 3) Wrap in MAML (with the same inner‑loop settings you used live)
maml_model = MAMLGraphTransformer(base_model, inner_lr=0.1, inner_steps=10)
maml_model.to(device).eval()

# 4) Run adaptation on the support set, then evaluate on the query set
adapted_params = maml_model.adapt(support_x, support_y)
logits_q       = functional_call(base_model, adapted_params, (query_x,))
preds_q        = logits_q.argmax(dim=1)
acc            = (preds_q == query_y).float().mean().item()

print(f"Few‑shot eval accuracy: {acc*100:.2f}%")
print("Predictions:", preds_q.cpu().numpy())
print("Ground truth:", query_y.cpu().numpy())
