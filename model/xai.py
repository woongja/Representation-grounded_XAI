"""
SC-XAI: Saliency-Constrained Explainable AI for Audio Deepfake Detection

Implementation with modular explainability losses for ablation studies.
Supports Model A-F configurations for comprehensive evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from .conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones


# ============================================================================
# Importance Generator
# ============================================================================
class ImportanceGenerator(nn.Module):
    """
    Generates temporal importance weights from SSL embeddings
    """
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(ImportanceGenerator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Importance in [0, 1]
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, T_f, D) - SSL encoder outputs
        Returns:
            importance: (B, T_f, 1) - temporal importance weights
        """
        importance = self.network(embeddings)  # (B, T_f, 1)
        return importance


# ============================================================================
# SSL Encoder Wrapper (XLSR)
# ============================================================================
class SSLEncoder(nn.Module):
    """
    Frozen SSL encoder (XLS-R)
    """
    def __init__(self, model_path='/home/woongjae/wildspoof/xlsr2_300m.pt', device='cuda'):
        super(SSLEncoder, self).__init__()

        # Load XLSR model
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024  # XLSR output dimension

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, waveform):
        """
        Args:
            waveform: (B, T_raw) - raw audio waveform
        Returns:
            embeddings: (B, T_f, 1024) - SSL features
        """
        # Ensure model is on correct device
        if next(self.model.parameters()).device != waveform.device:
            self.model.to(waveform.device, dtype=waveform.dtype)
            self.model.eval()

        # Extract features
        with torch.no_grad():
            result = self.model(waveform, mask=False, features_only=True)
            embeddings = result['x']  # (B, T_f, 1024)

        return embeddings


# ============================================================================
# Helper functions
# ============================================================================
def sinusoidal_embedding(n_channels, dim):
    """Sinusoidal positional embeddings"""
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)


# ============================================================================
# Conformer Classifier
# ============================================================================
class ConformerClassifier(nn.Module):
    """
    Conformer-based classifier for temporal features
    """
    def __init__(self, emb_size=144, heads=4, ffmult=4, exp_fac=2, kernel_size=31, n_encoders=4):
        super(ConformerClassifier, self).__init__()
        self.dim_head = int(emb_size / heads)
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders

        # Positional embeddings
        self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)

        # Conformer encoder blocks
        self.encoder_blocks = _get_clones(
            ConformerBlock(
                dim=emb_size,
                dim_head=self.dim_head,
                heads=heads,
                ff_mult=ffmult,
                conv_expansion_factor=exp_fac,
                conv_kernel_size=kernel_size
            ),
            n_encoders
        )

        # Class token
        self.class_token = nn.Parameter(torch.rand(1, emb_size))

        # Final classifier
        self.fc = nn.Linear(emb_size, 2)

    def forward(self, x, device):
        """
        Args:
            x: (B, T, emb_size) - temporal features
            device: torch device
        Returns:
            logits: (B, 2)
            embedding: (B, emb_size) - class token representation
        """
        # Add positional embeddings
        x = x + self.positional_emb[:, :x.size(1), :]

        # Add class token
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])  # (B, 1+T, emb_size)

        # Pass through Conformer blocks
        for layer in self.encoder_blocks:
            x, _ = layer(x)  # (B, 1+T, emb_size)

        # Extract class token representation
        embedding = x[:, 0, :]  # (B, emb_size)

        # Classification
        logits = self.fc(embedding)  # (B, 2)

        return logits, embedding


# ============================================================================
# Detector Head with Conformer
# ============================================================================
class DetectorHead(nn.Module):
    """
    Importance-weighted pooling + Conformer classifier
    """
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=144,
        num_classes=2,
        conformer_heads=4,
        conformer_encoders=4,
        conformer_kernel_size=31
    ):
        super(DetectorHead, self).__init__()

        # Projection to conformer embedding size
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(num_features=hidden_dim)
        self.selu = nn.SELU(inplace=True)

        # Conformer classifier
        self.conformer = ConformerClassifier(
            emb_size=hidden_dim,
            heads=conformer_heads,
            n_encoders=conformer_encoders,
            kernel_size=conformer_kernel_size
        )

    def forward(self, embeddings, importance):
        """
        Args:
            embeddings: (B, T_f, D)
            importance: (B, T_f, 1)
        Returns:
            logits: (B, num_classes)
            pooled: (B, hidden_dim) - class token representation
        """
        # Weighted importance (but we'll use full sequence for Conformer)
        # Apply importance as attention weights to embeddings
        importance_norm = importance / (importance.sum(dim=1, keepdim=True) + 1e-9)
        weighted_emb = embeddings * importance_norm  # (B, T_f, D)

        # Project to conformer dimension
        x = self.projection(weighted_emb)  # (B, T_f, hidden_dim)

        # Batch normalization (transpose for BN)
        x = x.transpose(1, 2)  # (B, hidden_dim, T_f)
        x = self.bn(x)
        x = self.selu(x)
        x = x.transpose(1, 2)  # (B, T_f, hidden_dim)

        # Conformer classification
        device = embeddings.device
        logits, pooled = self.conformer(x, device)  # (B, 2), (B, hidden_dim)

        return logits, pooled


# ============================================================================
# SC-XAI Model
# ============================================================================
class SCXAIModel(nn.Module):
    """
    Saliency-Constrained Explainable AI Model

    Supports ablation studies with modular explainability losses:
    - Model A: No explainability losses
    - Model B: Consistency only
    - Model C: Sensitivity only
    - Model D: Sparsity only
    - Model E: All three losses (full SC-XAI)
    - Model F: Full SC-XAI + LAM-ready embeddings
    """
    def __init__(
        self,
        ssl_model_path='/home/woongjae/wildspoof/xlsr2_300m.pt',
        ssl_dim=1024,
        importance_hidden_dim=256,
        detector_hidden_dim=144,  # Conformer embedding size
        num_classes=2,
        # Conformer parameters
        conformer_heads=4,
        conformer_encoders=4,
        conformer_kernel_size=31,
        # Explainability loss flags
        use_consistency_loss=False,
        use_sensitivity_loss=False,
        use_sparsity_loss=False,
        # Loss weights
        alpha_consistency=0.1,
        beta_sensitivity=0.1,
        gamma_sparsity=0.01,
        # Sensitivity loss parameters
        sensitivity_noise_std=0.1,
        sensitivity_spoof_weight=2.0,
        sensitivity_bona_weight=0.5,
        # LAM support
        return_lam_embedding=False,
        device='cuda'
    ):
        super(SCXAIModel, self).__init__()

        # Model configuration
        self.use_consistency_loss = use_consistency_loss
        self.use_sensitivity_loss = use_sensitivity_loss
        self.use_sparsity_loss = use_sparsity_loss

        self.alpha_consistency = alpha_consistency
        self.beta_sensitivity = beta_sensitivity
        self.gamma_sparsity = gamma_sparsity

        self.sensitivity_noise_std = sensitivity_noise_std
        self.sensitivity_spoof_weight = sensitivity_spoof_weight
        self.sensitivity_bona_weight = sensitivity_bona_weight

        self.return_lam_embedding = return_lam_embedding
        self.device = device

        # Components
        self.ssl_encoder = SSLEncoder(ssl_model_path, device)
        self.importance_generator = ImportanceGenerator(ssl_dim, importance_hidden_dim)
        self.detector_head = DetectorHead(
            input_dim=ssl_dim,
            hidden_dim=detector_hidden_dim,
            num_classes=num_classes,
            conformer_heads=conformer_heads,
            conformer_encoders=conformer_encoders,
            conformer_kernel_size=conformer_kernel_size
        )

        # Classification loss
        self.cls_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]))

    def forward(self, waveform, labels=None, debug=False):
        """
        Forward pass compatible with main.py

        Args:
            waveform: (B, T_raw) - raw audio waveform
            labels: (B,) - binary labels (0=fake, 1=real), optional
            debug: bool - print debug information

        Returns:
            outputs: dict containing:
                - logits: (B, num_classes)
                - embedding: (B, D) - for compatibility with main.py
                - importance: (B, T_f, 1) - importance map
                - ssl_emb: (B, T_f, D) - SSL embeddings (for LAM)
                - loss_total: total loss (if labels provided)
                - loss_cls: classification loss
                - loss_consistency: consistency loss (if enabled)
                - loss_sensitivity: sensitivity loss (if enabled)
                - loss_sparsity: sparsity loss (if enabled)
        """
        if debug:
            print("\n" + "="*80)
            print("SC-XAI Forward Pass - Debug Mode")
            print("="*80)
            print(f"Input waveform shape: {waveform.shape}")
            print(f"Explainability losses: Consistency={self.use_consistency_loss}, "
                  f"Sensitivity={self.use_sensitivity_loss}, Sparsity={self.use_sparsity_loss}")

        # 1. SSL Encoder (frozen)
        embeddings = self.ssl_encoder(waveform)  # (B, T_f, 1024)

        if debug:
            print(f"\n[1] SSL Embeddings: {embeddings.shape}")

        # 2. Importance Generator
        importance = self.importance_generator(embeddings)  # (B, T_f, 1)

        if debug:
            print(f"[2] Importance map: {importance.shape}")
            print(f"    Importance stats: min={importance.min().item():.4f}, "
                  f"max={importance.max().item():.4f}, mean={importance.mean().item():.4f}")

        # 3. Detector Head
        logits, pooled_emb = self.detector_head(embeddings, importance)

        if debug:
            print(f"[3] Classifier logits: {logits.shape}")
            print(f"    Pooled embedding: {pooled_emb.shape}")

        # Prepare outputs (compatible with main.py)
        outputs = {
            'logits': logits,
            'embedding': pooled_emb,  # For main.py compatibility
            'importance': importance,
            'ssl_emb': embeddings  # For LAM visualization
        }

        # 4. Compute losses if labels provided
        if labels is not None:
            loss_dict = self.compute_losses(
                logits=logits,
                importance=importance,
                embeddings=embeddings,
                labels=labels,
                debug=debug
            )
            outputs.update(loss_dict)

        if debug:
            print("\n" + "="*80)
            print("End of Forward Pass")
            print("="*80 + "\n")

        # return outputs
        return logits, pooled_emb

    def compute_losses(self, logits, importance, embeddings, labels, debug=False):
        """
        Compute all losses based on configuration

        Args:
            logits: (B, num_classes)
            importance: (B, T_f, 1)
            embeddings: (B, T_f, D)
            labels: (B,)
            debug: bool

        Returns:
            loss_dict: dict with all losses
        """
        loss_dict = {}

        # (1) Classification Loss
        loss_cls = self.cls_criterion(logits, labels)
        loss_dict['loss_cls'] = loss_cls

        if debug:
            print(f"\n[4] Losses:")
            print(f"    Classification: {loss_cls.item():.6f}")

        # Initialize total loss
        loss_total = loss_cls

        # (2) Consistency Loss
        if self.use_consistency_loss:
            loss_consistency = self.consistency_loss(importance)
            loss_dict['loss_consistency'] = loss_consistency
            loss_total = loss_total + self.alpha_consistency * loss_consistency

            if debug:
                print(f"    Consistency: {loss_consistency.item():.6f} (weight={self.alpha_consistency})")
        else:
            loss_dict['loss_consistency'] = None

        # (3) Sensitivity Loss
        if self.use_sensitivity_loss:
            loss_sensitivity = self.sensitivity_loss(embeddings, importance, labels)
            loss_dict['loss_sensitivity'] = loss_sensitivity
            loss_total = loss_total + self.beta_sensitivity * loss_sensitivity

            if debug:
                print(f"    Sensitivity: {loss_sensitivity.item():.6f} (weight={self.beta_sensitivity})")
        else:
            loss_dict['loss_sensitivity'] = None

        # (4) Sparsity Loss
        if self.use_sparsity_loss:
            loss_sparsity = self.sparsity_loss(importance)
            loss_dict['loss_sparsity'] = loss_sparsity
            loss_total = loss_total + self.gamma_sparsity * loss_sparsity

            if debug:
                print(f"    Sparsity: {loss_sparsity.item():.6f} (weight={self.gamma_sparsity})")
        else:
            loss_dict['loss_sparsity'] = None

        loss_dict['loss_total'] = loss_total

        if debug:
            print(f"    Total: {loss_total.item():.6f}")

        return loss_dict

    def consistency_loss(self, importance):
        """
        Temporal consistency loss
        Encourages smooth importance over time

        Args:
            importance: (B, T_f, 1)
        Returns:
            loss: scalar
        """
        # L_cons = mean(|w[:,1:] - w[:,:-1]|)
        diff = torch.abs(importance[:, 1:, :] - importance[:, :-1, :])
        loss = diff.mean()
        return loss

    def sensitivity_loss(self, embeddings, importance, labels):
        """
        Sensitivity loss
        Encourages robustness to small perturbations
        Different weights for spoof vs bonafide

        Args:
            embeddings: (B, T_f, D)
            importance: (B, T_f, 1)
            labels: (B,) - 0=fake, 1=real
        Returns:
            loss: scalar
        """
        # Add Gaussian noise to embeddings
        noise = torch.randn_like(embeddings) * self.sensitivity_noise_std
        embeddings_noisy = embeddings + noise

        # Compute importance for noisy embeddings
        with torch.set_grad_enabled(self.training):
            importance_noisy = self.importance_generator(embeddings_noisy)

        # Compute difference
        diff = torch.abs(importance - importance_noisy)  # (B, T_f, 1)
        diff_per_sample = diff.mean(dim=(1, 2))  # (B,)

        # Weight by label (spoof samples should be more robust)
        weights = torch.where(
            labels == 0,  # fake/spoof
            torch.tensor(self.sensitivity_spoof_weight, device=labels.device),
            torch.tensor(self.sensitivity_bona_weight, device=labels.device)
        )

        # Weighted loss
        loss = (diff_per_sample * weights).mean()

        return loss

    def sparsity_loss(self, importance):
        """
        Sparsity loss
        Encourages sparse (focused) importance

        Args:
            importance: (B, T_f, 1)
        Returns:
            loss: scalar
        """
        # L_sparse = mean(|w|)
        # Since importance is in [0,1], this encourages values close to 0
        # Flatten across time
        w = importance.view(importance.size(0), -1)  # (B, T_f)

        mean_w = w.mean(dim=1)        # (B,)
        std_w  = w.std(dim=1)         # (B,)

        # λ coefficient
        lambda_std = 0.5  # You can tune this later

        loss = (mean_w - lambda_std * std_w).mean()
        return loss


# ============================================================================
# Wrapper for main.py compatibility
# ============================================================================
class Model(SCXAIModel):
    """
    Wrapper class for compatibility with main.py training loop
    """
    def __init__(self, args, device):
        """
        Initialize SC-XAI model from args

        Args:
            args: argparse namespace with model config
            device: torch device
        """
        super(Model, self).__init__(
            ssl_model_path=getattr(args, 'ssl_model_path', '/home/woongjae/wildspoof/xlsr2_300m.pt'),
            ssl_dim=getattr(args, 'ssl_dim', 1024),
            importance_hidden_dim=getattr(args, 'importance_hidden_dim', 256),
            detector_hidden_dim=getattr(args, 'emb_size', 144),  # Use emb_size for conformer
            num_classes=getattr(args, 'num_classes', 2),
            conformer_heads=getattr(args, 'heads', 4),
            conformer_encoders=getattr(args, 'num_encoders', 4),
            conformer_kernel_size=getattr(args, 'kernel_size', 31),
            use_consistency_loss=getattr(args, 'use_consistency_loss', False),
            use_sensitivity_loss=getattr(args, 'use_sensitivity_loss', False),
            use_sparsity_loss=getattr(args, 'use_sparsity_loss', False),
            alpha_consistency=getattr(args, 'alpha_consistency', 0.1),
            beta_sensitivity=getattr(args, 'beta_sensitivity', 0.1),
            gamma_sparsity=getattr(args, 'gamma_sparsity', 0.01),
            sensitivity_noise_std=getattr(args, 'sensitivity_noise_std', 0.1),
            sensitivity_spoof_weight=getattr(args, 'sensitivity_spoof_weight', 2.0),
            sensitivity_bona_weight=getattr(args, 'sensitivity_bona_weight', 0.5),
            return_lam_embedding=getattr(args, 'return_lam_embedding', False),
            device=device
        )
