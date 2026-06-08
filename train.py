"""
MSCT-Net release script

This file provides a cleaned, reproducible implementation of the multi-scale
Transformer backbone, ECA recalibration, gated multi-scale fusion, LOSO training,
UF1/UAR evaluation, and computational-efficiency measurement.

Expected data format:
    data_root/
        subject_001/
            u_train/0/*.png
            u_train/1/*.png
            u_train/2/*.png
            u_test/0/*.png
            u_test/1/*.png
            u_test/2/*.png
        subject_002/
            ...

Class labels:
    0: Negative
    1: Positive
    2: Surprise
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.metrics import confusion_matrix
from torch import Tensor, einsum, nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def calculate_uf1_uar(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 3) -> Tuple[float, float]:
    """Calculate UF1 and UAR for class-imbalanced micro-expression recognition."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    f1_scores: List[float] = []
    recalls: List[float] = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        denom_f1 = 2 * tp + fp + fn
        denom_rec = tp + fn

        f1_scores.append((2 * tp / denom_f1) if denom_f1 > 0 else 0.0)
        recalls.append((tp / denom_rec) if denom_rec > 0 else 0.0)

    return float(np.mean(f1_scores)), float(np.mean(recalls))


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------


def get_eca_kernel_size(channels: int) -> int:
    """Adaptive ECA kernel size."""
    t = int(abs((math.log2(channels) + 1) / 2))
    return t if t % 2 == 1 else t + 1


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class ECAAttention(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels: int):
        super().__init__()
        kernel_size = get_eca_kernel_size(channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.gap(x)                         # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)       # (B, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y.expand_as(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [rearrange(t, "b (heads d) x y -> b heads (x y) d", heads=self.heads) for t in (q, k, v)]

        attn = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b heads (x y) d -> b (heads d) x y", x=h, y=w)
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, seq_len: int, depth: int, heads: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, dim, seq_len) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult=mlp_mult, dropout=dropout)),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        seq = h * w
        pos = self.pos_emb[:, :, :seq]
        pos = rearrange(pos, "1 c (h w) -> 1 c h w", h=h, w=w)
        x = x + pos
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class GatedFusion3(nn.Module):
    """Lightweight gated fusion for three aligned branch feature maps."""

    def __init__(self, dim: int, hidden: int = 128, n_branch: int = 3):
        super().__init__()
        self.n_branch = n_branch
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_branch * dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, n_branch, 1),
        )

    def forward(self, xs: Sequence[Tensor]) -> Tensor:
        if len(xs) != self.n_branch:
            raise ValueError(f"Expected {self.n_branch} feature maps, got {len(xs)}")
        x_cat = torch.cat(xs, dim=1)
        weights = self.gate(x_cat).flatten(1).softmax(dim=1)  # (B, 3)
        out = sum(x * weights[:, i].view(-1, 1, 1, 1) for i, x in enumerate(xs))
        return out


class MSCTNet(nn.Module):
    """Multi-scale Transformer network for three-class micro-expression recognition."""

    def __init__(
        self,
        image_size: int = 28,
        patch_sizes: Sequence[int] = (7, 14, 28),
        num_classes: int = 3,
        dim: int = 256,
        heads: int = 4,
        depths: Sequence[int] = (2, 2, 8),
        mlp_mult: int = 4,
        channels: int = 3,
        dropout: float = 0.1,
        use_eca: bool = True,
    ):
        super().__init__()
        if len(patch_sizes) != 3 or len(depths) != 3:
            raise ValueError("patch_sizes and depths must contain three values")
        for ps in patch_sizes:
            if image_size % ps != 0:
                raise ValueError(f"image_size={image_size} must be divisible by patch_size={ps}")

        fmap_sizes = [image_size // ps for ps in patch_sizes]
        self.target_size = fmap_sizes[1]  # medium-scale branch as reference

        self.branches = nn.ModuleList()
        for patch_size, fmap_size, depth in zip(patch_sizes, fmap_sizes, depths):
            patch_dim = channels * patch_size * patch_size
            patch_embed = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=patch_size, p2=patch_size),
                nn.Conv2d(patch_dim, dim, 1),
                LayerNorm2d(dim),
            )
            encoder = TransformerEncoder(dim, fmap_size * fmap_size, depth, heads, mlp_mult, dropout)
            eca = ECAAttention(dim) if use_eca else nn.Identity()
            self.branches.append(nn.ModuleList([patch_embed, encoder, eca]))

        self.gated_fusion = GatedFusion3(dim=dim, hidden=dim // 2, n_branch=3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features: List[Tensor] = []
        for patch_embed, encoder, eca in self.branches:
            feat = patch_embed(x)
            feat = encoder(feat)
            feat = eca(feat)
            if feat.shape[-2:] != (self.target_size, self.target_size):
                feat = F.interpolate(feat, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
            features.append(feat)

        fused = self.gated_fusion(features)
        pooled = self.global_pool(fused).flatten(1)
        return self.classifier(pooled)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_image(path: Path, image_size: int = 28) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # (C, H, W)


def load_split(split_dir: Path, image_size: int = 28) -> Tuple[Tensor, Tensor]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = int(class_dir.name)
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            xs.append(load_image(img_path, image_size=image_size))
            ys.append(label)

    if not xs:
        raise RuntimeError(f"No images found in {split_dir}")
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


def train_one_fold(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int, lr: float) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels.extend(y.tolist())
    return labels, preds


def run_loso(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_root = Path(args.data_root)

    all_subjects = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not all_subjects:
        raise RuntimeError(f"No subject folders found in {data_root}")

    total_gt: List[int] = []
    total_pred: List[int] = []
    start_time = time.time()

    for subject_dir in all_subjects:
        print(f"Subject: {subject_dir.name}")
        x_train, y_train = load_split(subject_dir / "u_train", image_size=args.image_size)
        x_test, y_test = load_split(subject_dir / "u_test", image_size=args.image_size)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

        model = MSCTNet(
            image_size=args.image_size,
            patch_sizes=tuple(args.patch_sizes),
            num_classes=args.num_classes,
            dim=args.dim,
            heads=args.heads,
            depths=tuple(args.depths),
            dropout=args.dropout,
            use_eca=not args.no_eca,
        ).to(device)

        train_one_fold(model, train_loader, device, epochs=args.epochs, lr=args.lr)
        gt, pred = predict(model, test_loader, device)
        total_gt.extend(gt)
        total_pred.extend(pred)

        uf1, uar = calculate_uf1_uar(total_gt, total_pred, num_classes=args.num_classes)
        print(f"Current UF1={uf1:.4f}, UAR={uar:.4f}")

    uf1, uar = calculate_uf1_uar(total_gt, total_pred, num_classes=args.num_classes)
    print("Final Evaluation")
    print(f"Total samples: {len(total_gt)}")
    print(f"UF1: {uf1:.4f}")
    print(f"UAR: {uar:.4f}")
    print(f"Total time: {time.time() - start_time:.2f} s")


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@torch.no_grad()
def measure_efficiency(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MSCTNet(
        image_size=args.image_size,
        patch_sizes=tuple(args.patch_sizes),
        num_classes=args.num_classes,
        dim=args.dim,
        heads=args.heads,
        depths=tuple(args.depths),
        dropout=args.dropout,
        use_eca=not args.no_eca,
    ).to(device).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    print(f"Params (M): {count_parameters(model):.4f}")

    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"FLOPs (G): {flops / 1e9:.4f}")
    except Exception as exc:
        print(f"FLOPs: unavailable ({exc}). Install thop to measure FLOPs.")

    # Warm-up
    for _ in range(20):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    for _ in range(args.runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) / args.runs * 1000
    print(f"Inference time per sample (ms): {elapsed_ms:.4f}")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"GPU memory (GB): {peak_memory:.4f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSCT-Net training and evaluation")
    parser.add_argument("--data_root", type=str, default="./data/three_norm_u_v_os")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--patch_sizes", type=int, nargs=3, default=[7, 14, 28])
    parser.add_argument("--depths", type=int, nargs=3, default=[2, 2, 8])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_eca", action="store_true")

    parser.add_argument("--measure_efficiency", action="store_true")
    parser.add_argument("--runs", type=int, default=500)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.measure_efficiency:
        measure_efficiency(args)
    else:
        run_loso(args)
