"""Shared behavior-cloning model, normalization, and metric helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from planning.contracts.feature import CONTINUOUS_DIM, NUM_ACTIONS, POLICY_VECTOR_DIM

def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        return torch, nn, F, Dataset, DataLoader
    except Exception as error:
        raise RuntimeError("PyTorch import failed: {}".format(repr(error)))

@dataclass
class VectorNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, continuous: np.ndarray, std_floor: float = 1.0e-3) -> "VectorNormalizer":
        values = np.asarray(continuous, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != CONTINUOUS_DIM:
            raise ValueError("continuous features must be [N,{}], got {}".format(CONTINUOUS_DIM, values.shape))
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, float(std_floor)).astype(np.float32)
        return cls(mean=mean, std=std)

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> "VectorNormalizer":
        if "feature_mean" not in checkpoint or "feature_std" not in checkpoint:
            raise KeyError("checkpoint has no feature normalization; retrain with the current feature contract")
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(CONTINUOUS_DIM)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(CONTINUOUS_DIM)
        return cls(mean=mean, std=std)

    def transform_continuous(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return ((array - self.mean) / np.maximum(self.std, 1.0e-6)).astype(np.float32)

def build_model(nn, vec_dim: int = POLICY_VECTOR_DIM, num_actions: int = NUM_ACTIONS, depth_channels: int = 1):
    if int(depth_channels) <= 0:
        raise ValueError("depth_channels must be positive")
    class BCPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_channels = int(depth_channels)
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(self.depth_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(inplace=True),
            )
            self.vector_encoder = nn.Sequential(
                nn.Linear(int(vec_dim), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 96),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Linear(128 + 96, 160),
                nn.ReLU(inplace=True),
                nn.Linear(160, int(num_actions)),
            )
        def forward(self, depth, vector):
            import torch
            depth_features = self.depth_encoder(depth)
            vector_features = self.vector_encoder(vector)
            return self.head(torch.cat([depth_features, vector_features], dim=1))

    return BCPolicy()

def mask_logits(logits, masks):
    return logits.masked_fill(~masks.bool(), -1.0e4)
