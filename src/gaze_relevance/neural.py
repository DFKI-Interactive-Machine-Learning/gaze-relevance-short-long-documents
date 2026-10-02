"""Neural models used in the GPU analyses (requires torch and torchvision, see requirements-gpu.txt).

VTNet (full, CNN branch only, GRU branch only) is the architecture used in the paper: a two-layer CNN with spatial
attention on the 150x150 grey-scale image and multi-head attention + GRU (hidden size 256) on the (x, y) time series,
fused by two fully connected layers.
VGG19 uses ImageNet-1K weights with a new output layer, 256x256 RGB input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        return x * torch.sigmoid(self.conv1(x))


class _CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.spatial_attention = SpatialAttention()
        self.fc1 = nn.Linear(16 * 34 * 34, 50)

    def forward(self, im):
        x = self.pool(nn.functional.relu(self.conv1(im)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.spatial_attention(x).view(-1, 16 * 34 * 34)
        return nn.functional.relu(self.fc1(x))


class _GRU(nn.Module):
    hidden_size = 256

    def __init__(self):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=2, num_heads=1)
        self.rnn = nn.GRU(input_size=2, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

    def forward(self, ts, lengths):
        x = ts.permute(1, 0, 2).float()
        x, _ = self.multihead_attn1(x, x, x, need_weights=False)
        x = x.permute(1, 0, 2)
        h0 = torch.zeros(1, ts.size(0), self.hidden_size, device=ts.device)
        out, _ = self.rnn(pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False), h0)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out[torch.arange(ts.size(0)), lengths.to(out.device) - 1]


class VTNet(nn.Module):
    """branches: 'full' (CNN + GRU), 'cnn_only', 'gru_only'. Returns one logit per sample."""

    def __init__(self, branches="full"):
        super().__init__()
        self.branches = branches
        self.cnn = _CNN() if branches in ("full", "cnn_only") else None
        self.gru = _GRU() if branches in ("full", "gru_only") else None
        width = {"full": 50 + 256, "cnn_only": 50, "gru_only": 256}[branches]
        self.fc2 = nn.Linear(width, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, im, ts, lengths):
        parts = []
        if self.cnn is not None:
            parts.append(self.cnn(im))
        if self.gru is not None:
            parts.append(self.gru(ts, lengths))
        return self.fc3(nn.functional.relu(self.fc2(torch.cat(parts, 1)))).squeeze(1)


class VTNetDataset(Dataset):
    """index: data frame with img_path, csv_path, label (as written by representations.generate)."""
    tf = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __init__(self, index: pd.DataFrame):
        self.images = [self.tf(Image.open(p).convert("L")) for p in index.img_path]
        self.series = [torch.tensor(pd.read_csv(p)[["x", "y"]].fillna(0).values, dtype=torch.float32) for p in index.csv_path]
        self.labels = index.label.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.series[i], self.labels[i]


def collate(batch):
    im, ts, y = zip(*batch)
    return torch.stack(im), pad_sequence(ts, batch_first=True), torch.tensor(y), torch.tensor([len(s) for s in ts])


def train_predict_vtnet(ds, tr, te, branches="full", seed=0, epochs=30, lr=1e-3, batch_size=16, device="cuda"):
    """Fixed hyperparameters (no tuning) so that all splits are comparable. Returns P(relevant) for the test trials."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VTNet(branches).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for im, ts, y, L in DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=True, collate_fn=collate):
            opt.zero_grad()
            lossf(model(im.to(device), ts.to(device), L), y.float().to(device)).backward()
            opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for im, ts, y, L in DataLoader(Subset(ds, te), batch_size=4, shuffle=False, collate_fn=collate):
            out.append(torch.sigmoid(model(im.to(device), ts.to(device), L)).cpu())
    return torch.cat(out).numpy()


VGG_TF = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def load_images_rgb(paths):
    return torch.stack([VGG_TF(Image.open(p).convert("RGB")) for p in paths])


def train_predict_vgg19(X, y, tr, te, n_classes, epochs=15, lr=1e-4, batch_size=16, device="cuda", seed=0):
    """VGG19 (ImageNet-1K) with a new output layer, class-weighted cross-entropy, fixed hyperparameters."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    m.classifier[6] = nn.Linear(4096, n_classes)
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    w = torch.tensor(np.bincount(y[tr], minlength=n_classes), dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=(w.sum() / (n_classes * w.clamp(min=1))).to(device))
    yt = torch.tensor(y)
    for _ in range(epochs):
        m.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            lossf(m(X[b].to(device)), yt[b].to(device)).backward()
            opt.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 32):
            out.append(m(X[te[i:i + 32]].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()
