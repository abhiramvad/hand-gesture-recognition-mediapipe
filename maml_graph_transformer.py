# maml_graph_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict

class MAMLGraphTransformer(nn.Module):
    def __init__(self, base_model, inner_lr=0.01, inner_steps=1):
        super().__init__()
        self.base_model  = base_model
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps

    def adapt(self, support_x, support_y):
        # start from base_model parameters
        adapted_params = OrderedDict(self.base_model.named_parameters())
        for _ in range(self.inner_steps):
            logits = functional_call(self.base_model, adapted_params, (support_x,))
            loss   = F.cross_entropy(logits, support_y)
            grads  = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            adapted_params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(adapted_params.items(), grads)
            )
        return adapted_params

    def forward(self, tasks):
        meta_loss = 0.0
        meta_acc  = 0.0
        for support_x, support_y, query_x, query_y in tasks:
            adapted = self.adapt(support_x, support_y)
            q_logits = functional_call(self.base_model, adapted, (query_x,))
            loss_q   = F.cross_entropy(q_logits, query_y)
            acc_q    = (q_logits.argmax(dim=1) == query_y).float().mean()
            meta_loss += loss_q
            meta_acc  += acc_q
        meta_loss /= len(tasks)
        meta_acc  /= len(tasks)
        return meta_loss, meta_acc

    def meta_update(self, support_x, support_y, query_x, query_y):
        adapted  = self.adapt(support_x, support_y)
        q_logits = functional_call(self.base_model, adapted, (query_x,))
        loss_q   = F.cross_entropy(q_logits, query_y)
        acc_q    = (q_logits.argmax(dim=1) == query_y).float().mean()
        return loss_q, acc_q
