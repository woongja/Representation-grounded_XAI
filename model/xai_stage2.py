"""
STAGE 2: XAI Module on Frozen Detector

This module:
1. Loads a pretrained detector (conformertcm)
2. Freezes ALL detector parameters
3. Adds XAI module (Importance Network + Prototypes)
4. Trains ONLY the XAI module

Compatible with existing main.py infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# ============================================================================
# Frozen Detector Wrapper
# ============================================================================
class FrozenDetector(nn.Module):
    """
    Loads and freezes a pretrained detector.
    Extracts SSL embeddings for XAI module.
    """

    def __init__(self, checkpoint_path, args, device):
        super(FrozenDetector, self).__init__()

        # Load pretrained detector
        print(f"\n{'='*80}")
        print(f"Loading pretrained detector from: {checkpoint_path}")

        # Import conformertcm model
        from .conformertcm import Model as DetectorModel

        self.detector = DetectorModel(args, device)

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.detector.load_state_dict(state_dict, strict=False)

        print(f"✓ Detector loaded successfully")

        # Freeze ALL parameters
        for param in self.detector.parameters():
            param.requires_grad = False

        self.detector.eval()

        frozen_params = sum(p.numel() for p in self.detector.parameters())
        print(f"✓ Frozen {frozen_params:,} parameters")
        print(f"{'='*80}\n")

    def forward(self, x):
        """
        Extract SSL embeddings from frozen detector.

        Args:
            x: (B, T_raw) waveform

        Returns:
            embeddings: (B, T_frame, 1024) SSL features
        """
        with torch.no_grad():
            # Extract SSL features
            # conformertcm: x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
            embeddings = self.detector.ssl_model.extract_feat(x.squeeze(-1))

        return embeddings


# ============================================================================
# XAI Components
# ============================================================================
class ImportanceNetwork(nn.Module):
    """Generates frame-level importance weights."""

    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        """h: (B, T, D) -> w: (B, T, 1)"""
        return self.net(h)


class PrototypeManager(nn.Module):
    """
    Manages class prototypes.

    Modes:
    - 'ema': EMA update (default, most stable)
    - 'fixed': Global mean (ablation)
    - 'learnable': Backprop (fallback)
    """

    def __init__(
        self,
        dim=1024,
        num_classes=2,
        mode='ema',
        momentum=0.99,
        normalize=True
    ):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.mode = mode
        self.momentum = momentum
        self.normalize = normalize

        # Initialize prototypes
        if mode == 'learnable':
            self.prototypes = nn.Parameter(torch.randn(num_classes, dim))
        else:
            self.register_buffer('prototypes', torch.randn(num_classes, dim))

        # Normalize
        if normalize:
            with torch.no_grad():
                self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def update_ema(self, embeddings, labels, importance=None):
        """Update prototypes via EMA."""
        if self.mode != 'ema':
            return

        with torch.no_grad():
            B, T, D = embeddings.shape

            # Flatten
            h_flat = embeddings.view(-1, D)
            labels_flat = labels.unsqueeze(1).expand(B, T).reshape(-1)

            # Update each class
            for c in range(self.num_classes):
                mask = (labels_flat == c)
                if mask.sum() == 0:
                    continue

                h_c = h_flat[mask]
                batch_mean = h_c.mean(dim=0)

                # EMA: p ← m*p + (1-m)*batch_mean
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c] +
                    (1 - self.momentum) * batch_mean
                )

            # Normalize
            if self.normalize:
                self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def compute_scores(self, embeddings):
        """
        Compute frame-level similarity to prototypes.

        Args:
            embeddings: (B, T, D)

        Returns:
            scores: (B, T, C) where scores[b,t,c] = h(t)·p_c
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        # (B, T, D) @ (D, C) -> (B, T, C)
        scores = torch.matmul(embeddings, self.prototypes.t())
        return scores

    def forward(self, embeddings):
        return self.compute_scores(embeddings)


# ============================================================================
# STAGE 2: Complete XAI Model
# ============================================================================
class Model(nn.Module):
    """
    Stage 2 XAI Model

    Compatible with main.py training infrastructure.
    """

    def __init__(self, args, device):
        super().__init__()

        self.device = device

        # Get checkpoint path
        checkpoint_path = getattr(
            args,
            'pretrained_checkpoint',
            '/home/woongjae/ADD_LAB/Representation-grounded_XAI/avg_5_best.pth'
        )

        # XAI configuration
        importance_hidden = getattr(args, 'importance_hidden_dim', 256)
        prototype_mode = getattr(args, 'prototype_mode', 'ema')
        ema_momentum = getattr(args, 'ema_momentum', 0.99)
        self.temperature = getattr(args, 'temperature', 0.07)

        # Loss weights
        self.lambda_bonafide = getattr(args, 'lambda_bonafide_reg', 0.1)
        self.lambda_smooth = getattr(args, 'lambda_temporal_smooth', 0.1)
        self.lambda_sparsity = getattr(args, 'lambda_sparsity', 0.01)

        print(f"\n{'='*80}")
        print("STAGE 2: Representation-Based XAI")
        print(f"{'='*80}")
        print(f"  Pretrained detector: {checkpoint_path}")
        print(f"  Prototype mode: {prototype_mode.upper()}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Loss weights:")
        print(f"    - Bonafide reg: {self.lambda_bonafide}")
        print(f"    - Temporal smooth: {self.lambda_smooth}")
        print(f"    - Sparsity: {self.lambda_sparsity}")

        # ========================================
        # FROZEN DETECTOR
        # ========================================
        self.frozen_detector = FrozenDetector(checkpoint_path, args, device)

        # ========================================
        # TRAINABLE XAI MODULE
        # ========================================
        self.importance_net = ImportanceNetwork(
            input_dim=1024,
            hidden_dim=importance_hidden
        )

        self.prototype_manager = PrototypeManager(
            dim=1024,
            num_classes=2,
            mode=prototype_mode,
            momentum=ema_momentum,
            normalize=True
        )

        # Print trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable

        print(f"\n  Parameters:")
        print(f"    - Total: {total:,}")
        print(f"    - Frozen (Detector): {frozen:,}")
        print(f"    - Trainable (XAI): {trainable:,}")
        print(f"{'='*80}\n")

    def forward(self, waveform, labels=None):
        """
        Forward pass compatible with main.py.

        Args:
            waveform: (B, T_raw)
            labels: (B,) optional

        Returns:
            logits: (B, 2) - for compatibility with main.py
            pooled_emb: (B, D) - for compatibility
        """
        # Extract frozen embeddings
        embeddings = self.frozen_detector(waveform)  # (B, T, 1024)

        # Generate importance
        importance = self.importance_net(embeddings)  # (B, T, 1)

        # Compute prototype scores
        frame_scores = self.prototype_manager(embeddings)  # (B, T, 2)

        # Aggregate to utterance level (for main.py compatibility)
        importance_norm = importance / (importance.sum(dim=1, keepdim=True) + 1e-9)
        utterance_scores = (frame_scores * importance_norm).sum(dim=1)  # (B, 2)

        # Weighted embedding
        pooled_emb = (embeddings * importance_norm).sum(dim=1)  # (B, 1024)

        # Compute loss if labels provided (for both train and eval)
        if labels is not None:
            self.compute_loss(embeddings, importance, frame_scores, labels)

        return utterance_scores, pooled_emb

    def compute_loss(self, embeddings, importance, frame_scores, labels):
        """
        Compute XAI losses.

        This modifies the standard CrossEntropyLoss used in main.py.
        """
        B, T, C = frame_scores.shape

        # ========================================
        # 1. Frame-level Contrastive Loss
        # ========================================
        frame_scores_scaled = frame_scores / self.temperature
        labels_expanded = labels.unsqueeze(1).expand(B, T)

        log_probs = F.log_softmax(frame_scores_scaled, dim=-1)
        frame_loss = -log_probs.gather(dim=-1, index=labels_expanded.unsqueeze(-1))
        frame_loss = frame_loss.squeeze(-1)  # (B, T)

        # Importance-weighted
        importance_sq = importance.squeeze(-1)
        contrastive_loss = (importance_sq * frame_loss).sum(dim=1).mean()

        # ========================================
        # 2. Regularization
        # ========================================
        # Bonafide: low importance
        bonafide_mask = (labels == 1)
        if bonafide_mask.sum() > 0:
            bonafide_reg = importance_sq[bonafide_mask].sum(dim=1).mean()
        else:
            bonafide_reg = torch.tensor(0.0, device=embeddings.device)

        # Temporal smoothness
        smooth_reg = torch.abs(importance_sq[:, 1:] - importance_sq[:, :-1]).mean()

        # Sparsity (spoof)
        spoof_mask = (labels == 0)
        if spoof_mask.sum() > 0:
            sparsity_reg = importance_sq[spoof_mask].abs().mean()
        else:
            sparsity_reg = torch.tensor(0.0, device=embeddings.device)

        # Total loss (stored for retrieval by main.py)
        self.last_loss = (
            contrastive_loss +
            self.lambda_bonafide * bonafide_reg +
            self.lambda_smooth * smooth_reg +
            self.lambda_sparsity * sparsity_reg
        )

        # Update EMA prototypes
        if self.prototype_manager.mode == 'ema':
            self.prototype_manager.update_ema(embeddings, labels, importance)

        return self.last_loss
