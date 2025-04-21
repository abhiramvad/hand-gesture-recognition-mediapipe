# meta_training_graph.py

import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from maml_graph_transformer import MAMLGraphTransformer
from graph_transformer      import GraphTransformerClassifier

def create_few_shot_task(X, y, n_way, k_shot, q_query):
    classes  = torch.unique(y)
    selected = classes[torch.randperm(len(classes))[:n_way]]
    support_x, support_y, query_x, query_y = [], [], [], []
    for cls in selected:
        idx = (y == cls).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx))]
        s_idx, q_idx = idx[:k_shot], idx[k_shot:k_shot + q_query]
        support_x.append(X[s_idx]); support_y.append(y[s_idx])
        query_x.append(X[q_idx]);   query_y.append(y[q_idx])
    return (torch.cat(support_x), torch.cat(support_y),
            torch.cat(query_x),   torch.cat(query_y))

def train_meta(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load & split
    data = np.loadtxt(args.data_path, delimiter=',', dtype=np.float32)
    X    = torch.tensor(data[:,1:], dtype=torch.float32).view(-1,21,2)
    y    = torch.tensor(data[:,0],   dtype=torch.long)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=args.seed
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(torch.unique(y).size(0))

    base_model = GraphTransformerClassifier(
        input_dim=2, hidden_dim=args.hidden_dim, num_classes=num_classes
    ).to(device)
    maml = MAMLGraphTransformer(
        base_model, inner_lr=args.inner_lr, inner_steps=args.inner_steps
    ).to(device)

    optimizer    = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- meta‑train ---
        maml.train()
        tasks = []
        for _ in range(args.meta_batch_size):
            s_x, s_y, q_x, q_y = create_few_shot_task(
                X_train, y_train,
                args.n_way, args.k_shot, args.q_query
            )
            tasks.append((s_x.to(device), s_y.to(device),
                          q_x.to(device), q_y.to(device)))
        meta_loss, meta_acc = maml(tasks)
        optimizer.zero_grad(); meta_loss.backward(); optimizer.step()

        # --- validation ---
        maml.eval()
        with torch.no_grad():
            vs_x, vs_y, vq_x, vq_y = create_few_shot_task(
                X_val, y_val,
                args.n_way, args.k_shot, args.q_query
            )
            vs_x, vs_y = vs_x.to(device), vs_y.to(device)
            vq_x, vq_y = vq_x.to(device), vq_y.to(device)
        val_loss, val_acc = maml.meta_update(vs_x, vs_y, vq_x, vq_y)

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"Train Loss: {meta_loss.item():.4f}, Train Acc: {meta_acc.item():.4f} | "
              f"Val Loss:   {val_loss.item():.4f}, Val Acc:   {val_acc.item():.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(base_model.state_dict(), args.save_path)
            print(f"✅ Saved best model with Val Acc: {val_acc:.4f}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",       type=str,   default="model/keypoint_classifier/keypoint.csv")
    p.add_argument("--save_path",       type=str,   default="model/keypoint_classifier/maml_graph_transformer.pt")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--meta_batch_size", type=int,   default=4)
    p.add_argument("--n_way",           type=int,   default=5)
    p.add_argument("--k_shot",          type=int,   default=5)
    p.add_argument("--q_query",         type=int,   default=15)
    p.add_argument("--meta_lr",         type=float, default=0.001)
    p.add_argument("--inner_lr",        type=float, default=0.01)
    p.add_argument("--inner_steps",     type=int,   default=1)
    p.add_argument("--hidden_dim",      type=int,   default=64)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_meta(args)
