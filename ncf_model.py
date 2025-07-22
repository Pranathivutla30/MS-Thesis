# ncf_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from preprocess import preprocess_and_split

# Dataset
class NCFDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor(df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x)

# Discriminator for Adversarial Debiasing
class Discriminator(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, user_embedding):
        return self.model(user_embedding)

# IRM Penalty
def irm_penalty(logits, labels):
    scale = torch.tensor(1.0, requires_grad=True)
    loss = nn.BCELoss()(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

# Evaluation Functions
def precision_at_k(y_true, y_pred, k=5):
    precisions = []
    for true, pred in zip(y_true, y_pred):
        top_k = np.argsort(pred)[::-1][:k]
        precisions.append(np.sum(true[top_k]) / k)
    return np.mean(precisions)

def ndcg_at_k(y_true, y_pred, k=5):
    ndcgs = []
    for true, pred in zip(y_true, y_pred):
        ndcgs.append(ndcg_score([true], [pred], k=k))
    return np.mean(ndcgs)

# Train + Evaluate
def train_and_evaluate():
    train_df, test_df, num_users, num_items = preprocess_and_split()
    train_loader = DataLoader(NCFDataset(train_df), batch_size=128, shuffle=True)
    test_loader = DataLoader(NCFDataset(test_df), batch_size=128, shuffle=False)

    model = NCF(num_users, num_items)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for user, item, label in train_loader:
            optimizer.zero_grad()

            user_emb = model.user_embedding(user)
            item_emb = model.item_embedding(item)
            x = torch.cat([user_emb, item_emb], dim=-1)
            pred = model.mlp(x).squeeze()

            # IRM loss across environments
            env1_idx = torch.randperm(len(user))[:len(user)//2]
            env2_idx = torch.randperm(len(user))[len(user)//2:]
            pred1, label1 = pred[env1_idx], label[env1_idx]
            pred2, label2 = pred[env2_idx], label[env2_idx]
            loss1 = criterion(pred1, label1)
            loss2 = criterion(pred2, label2)
            penalty1 = irm_penalty(pred1, label1)
            penalty2 = irm_penalty(pred2, label2)
            irm_loss = (loss1 + loss2)/2 + 1.0 * (penalty1 + penalty2)/2

            # Adversarial loss (using mapped batch user IDs)
            unique_users, mapped_user = torch.unique(user, return_inverse=True)
            discriminator = Discriminator(embedding_dim=32, num_classes=len(unique_users))
            disc_logits = discriminator(user_emb.detach())
            disc_loss = nn.CrossEntropyLoss()(disc_logits, mapped_user)

            fool_logits = discriminator(user_emb)
            gen_adv_loss = -nn.CrossEntropyLoss()(fool_logits, mapped_user)

            # Total loss
            total_gen_loss = irm_loss + 1.0 * gen_adv_loss
            total_gen_loss.backward()
            optimizer.step()

            disc_loss.backward()
            optimizer.step()

            total_loss += total_gen_loss.item()
        print(f"Epoch {epoch+1} - IRM+Adv Loss: {total_loss:.4f}")

    model.eval()
    user_item_matrix = {}
    with torch.no_grad():
        for user, item, label in test_loader:
            output = model(user, item).squeeze()
            for u, i, l, o in zip(user, item, label, output):
                u = u.item()
                if u not in user_item_matrix:
                    user_item_matrix[u] = {'y_true': [], 'y_pred': []}
                user_item_matrix[u]['y_true'].append(1)
                user_item_matrix[u]['y_pred'].append(o.item())

    y_true, y_pred = [], []
    for v in user_item_matrix.values():
        y_true.append(np.array(v['y_true'] + [0]*5))
        y_pred.append(np.array(v['y_pred'] + list(np.random.rand(5))))

    print("\n Evaluation:")
    print(f"Precision@5: {precision_at_k(y_true, y_pred):.4f}")
    print(f"NDCG@5: {ndcg_at_k(y_true, y_pred):.4f}")

if __name__ == "__main__":
    train_and_evaluate()
