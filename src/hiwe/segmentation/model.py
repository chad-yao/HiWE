"""Semi-supervised key interval segmentation model (ResNet + GRU)."""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision.transforms.functional import to_tensor, resize


class FocalLoss(nn.Module):
    """Focal loss for class imbalance in key interval classification."""

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = at * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class GaussianNoise(nn.Module):
    """Gaussian noise augmentation for robustness."""

    def __init__(self, std=0.5):
        super().__init__()
        self.std = std

    def forward(self, tensor):
        if self.training:
            noise = torch.randn_like(tensor) * self.std
            return tensor + noise
        return tensor


class KeyIntervalSegmenter(nn.Module):
    """ResNet-18 + GRU model for key interval segmentation from image sequences."""

    def __init__(
        self,
        hidden_size=256,
        num_layers=1,
        bidirectional=True,
        noise_std=0.5,
        is_semi=False,
    ):
        super().__init__()
        self.gaussian_noise = GaussianNoise(std=noise_std)
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.num_layers = num_layers
        self.is_semi = is_semi
        self.resnet.fc = nn.Identity()
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, 2
        )

    def forward(self, x, hidden=None):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        x = self.gaussian_noise(x)
        x = x.view(batch_size, seq_len, C, H, W)
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.resnet(x)
        features = features.view(batch_size, seq_len, -1)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        gru_out, hidden = self.gru(features, hidden)
        gru_out = self.dropout(gru_out)
        gru_out = gru_out.reshape(-1, gru_out.shape[2])
        out = self.fc(gru_out)
        out = out.view(batch_size, seq_len, -1)
        return out, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(
            self.num_layers * 2, batch_size, 256, device=device
        )


class ImageSequenceHDF5Dataset(Dataset):
    """Dataset for image sequences from HDF5 files (ACT or DP format)."""

    def __init__(
        self,
        hdf5_root_path,
        indice,
        seq_len=None,
        transform=None,
        semi=False,
        size=112,
        labelset=None,
        image_key="/observations/image/agent",
    ):
        self.hdf5_files = [
            os.path.join(hdf5_root_path, f"episode_{i}.hdf5") for i in indice
        ]
        self.seq_len = seq_len
        self.transform = transform
        self.indices = []
        self.size = size
        self.semi = semi
        self.labelset = set(labelset) if labelset else set()
        self.image_key = image_key

        for file_idx, file_path in enumerate(self.hdf5_files):
            with h5py.File(file_path, "r") as file:
                total_frames = file[self.image_key].shape[0]
                if self.seq_len is None:
                    self.seq_len = total_frames
                num_sequences = total_frames // self.seq_len
                for seq_idx in range(num_sequences + 1):
                    start_idx = seq_idx * self.seq_len
                    if start_idx < total_frames:
                        self.indices.append((file_idx, start_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start_idx = self.indices[idx]
        with h5py.File(self.hdf5_files[file_idx], "r") as file:
            mid_images = file[self.image_key][
                start_idx : start_idx + self.seq_len
            ]
            if self.semi and file_idx not in self.labelset:
                labels = torch.full(
                    (self.seq_len,), -1, dtype=torch.long
                )
            else:
                labels = file["/label"][start_idx : start_idx + self.seq_len]
                labels = torch.tensor(labels, dtype=torch.long)
            transformed_images = []
            for image in mid_images:
                image_tensor = to_tensor(image)
                if self.transform:
                    image_tensor = self.transform(image_tensor)
                else:
                    image_tensor = resize(
                        image_tensor, (self.size, self.size), antialias=True
                    )
                transformed_images.append(image_tensor)
            images = torch.stack(transformed_images)
        return images, labels, file_idx, start_idx
