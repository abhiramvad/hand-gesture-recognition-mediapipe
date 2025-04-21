import torch
import torch.nn as nn
import torch.nn.functional as F 

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=11, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        h = self.transformer_encoder(h)
        h = h.mean(dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


class MAMLTransformer(nn.Module):
    def __init__(self, base_model, inner_lr=0.01, inner_steps=1):
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def forward(self, tasks):
        meta_loss = 0
        meta_acc = 0

        for support_x, support_y, query_x, query_y in tasks:
            fast_weights = list(self.base_model.parameters())

            for _ in range(self.inner_steps):
                outputs = self.base_model(support_x)
                loss = F.cross_entropy(outputs, support_y)
                grads = torch.autograd.grad(loss, self.base_model.parameters(), create_graph=True)
                fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]

            # Apply fast weights manually to query set
            def forward_with_weights(x, weights):
                h = F.linear(x, weights[0], weights[1])  # input_proj
                h = F.relu(h)
                h = self.base_model.transformer_encoder(h)
                h = h.mean(dim=1)
                h = F.linear(F.dropout(F.relu(F.linear(h, weights[4], weights[5])), p=0.2), weights[6], weights[7])
                return h

            outputs_q = forward_with_weights(query_x, fast_weights)
            loss_q = F.cross_entropy(outputs_q, query_y)
            acc_q = (outputs_q.argmax(dim=1) == query_y).float().mean()

            meta_loss += loss_q
            meta_acc += acc_q

        return meta_loss / len(tasks), meta_acc / len(tasks)

    
    def meta_update(self, support_x, support_y, query_x, query_y, inner_steps=1):
        fast_weights = list(self.base_model.parameters())

        for _ in range(inner_steps):
            outputs = self.base_model(support_x)
            loss = F.cross_entropy(outputs, support_y)
            grads = torch.autograd.grad(loss, self.base_model.parameters(), create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]

        # Apply fast weights manually to query set
        def forward_with_weights(x, weights):
            h = F.linear(x, weights[0], weights[1])
            h = F.relu(h)
            h = self.base_model.transformer_encoder(h)
            h = h.mean(dim=1)
            h = F.linear(F.dropout(F.relu(F.linear(h, weights[4], weights[5])), p=0.2), weights[6], weights[7])
            return h

        query_preds = forward_with_weights(query_x, fast_weights)
        loss_q = F.cross_entropy(query_preds, query_y)
        acc_q = (query_preds.argmax(dim=1) == query_y).float().mean()

        return loss_q, acc_q

