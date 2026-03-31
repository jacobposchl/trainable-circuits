"""
Phase 1 Trainer: Meta-Encoder Validation.

Trains a meta-encoder to learn circuit-space representations from a mostly
frozen backbone's activation trajectories. By default the backbone stays frozen,
but configs may opt into training the CIFAR-adapted ResNet stem conv.

    L_total = L_info

where:
  L_info:  Flow co-activation reconstruction fidelity, normalized as (1 - R²):
           SS_res_l / SS_tot_l  where SS_res = ||MLP_l(z_l^a * z_l^b) - f_l^a ⊙ f_l^b||²_F
           Scale-invariant: ~1.0 at init, → 0 at perfect reconstruction.
           f_l(x) = compressed, L2-normalised bn2/bn3 output (pre-skip),
           isolating the pure block contribution from accumulated history.

All pairs are formed within-batch.  No class-label pairing needed — the
training signal comes entirely from flow co-activation profiles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.backbone import FrozenBackbone
from models.meta_encoder import MetaEncoder
from losses.info_loss import InfoLoss
from data.cifar import get_standard_loaders


class Phase1Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_val_metrics: dict = {}

        self._build_models()
        self._build_data()
        self._build_losses()      # must come before _build_optimizers
        self._build_optimizers()

        self.checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _build_models(self):
        mcfg = self.cfg["model"]
        fcfg = mcfg.get("flow_compression", {})

        self.backbone = FrozenBackbone(
            arch=mcfg["arch"],
            num_classes=mcfg.get("num_classes", 10),
            pretrained=mcfg.get("pretrained", True),
            grid_size=fcfg.get("grid_size", 4),
            flow_dim=fcfg.get("flow_dim", 256),
            trainable_stem=mcfg.get("trainable_stem", False),
        ).to(self.device)

        ecfg = mcfg["meta_encoder"]
        self.meta_encoder = MetaEncoder(
            layer_dims=self.backbone.layer_dims,   # [D_flow] * L
            projection_dim=ecfg.get("projection_dim", 128),
            n_heads=ecfg.get("n_heads", 4),
            n_transformer_layers=ecfg.get("n_transformer_layers", 2),
            dropout=ecfg.get("dropout", 0.0),
        ).to(self.device)

    def _build_data(self):
        dcfg = self.cfg["data"]
        self.train_loader, self.val_loader = get_standard_loaders(
            data_dir=dcfg.get("data_dir", "data/cifar10"),
            batch_size=dcfg.get("batch_size", 256),
            num_workers=dcfg.get("num_workers", 4),
            augment=dcfg.get("augment", True),
            download=True,
        )

    def _build_optimizers(self):
        tcfg = self.cfg["training"]
        lr = float(tcfg.get("lr", 1e-3))
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        params = (
            backbone_params
            + list(self.meta_encoder.parameters())
            + list(self.info_loss.parameters())
        )
        self.optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg["training"]["epochs"],
            eta_min=lr * 0.01,
        )

    def _build_losses(self):
        mcfg = self.cfg["model"]
        tcfg = self.cfg["training"]
        ecfg = mcfg["meta_encoder"]
        rcfg = mcfg.get("regressor", {})

        self.info_loss = InfoLoss(
            layer_dims=self.backbone.layer_dims,   # [D_flow] * L
            projection_dim=ecfg.get("projection_dim", 128),
            hidden_dim=rcfg.get("hidden_dim", 64),
        ).to(self.device)

        self.info_loss_weight = float(tcfg.get("info_loss_weight", 5.0))

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(self, resume_from: str | None = None):
        start_epoch = 0
        best_val_r2 = -float("inf")
        tcfg = self.cfg["training"]
        patience = tcfg.get("early_stopping_patience")
        min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
        epochs_without_improvement = 0

        if resume_from is not None:
            start_epoch = self._load_checkpoint(resume_from)
            best_val_r2 = float(self._loaded_val_metrics.get("r2", best_val_r2))

        epochs       = self.cfg["training"]["epochs"]
        log_interval = self.cfg["logging"].get("log_interval", 50)
        save_every   = self.cfg["logging"].get("save_every", 10)

        for epoch in range(start_epoch, epochs):
            train_metrics = self._train_epoch(epoch, log_interval)
            val_metrics   = self._val_epoch()
            self.lr_scheduler.step()

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f} "
                f"info={train_metrics['info_loss']:.2e} | "
                f"val_R2={val_metrics['r2']:.3f} "
                f"val_rho={val_metrics['mean_rho']:.3f}"
            )

            is_best = val_metrics["r2"] > (best_val_r2 + min_delta)
            if is_best:
                best_val_r2 = val_metrics["r2"]
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics, name="best.pt")
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, val_metrics, name=f"epoch_{epoch+1}.pt")

            if patience is not None and patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch+1}: "
                    f"val_R2 has not improved by at least {min_delta:.4f} "
                    f"for {patience} epoch(s)."
                )
                break

    def _train_epoch(self, epoch: int, log_interval: int) -> dict:
        self.meta_encoder.train()
        self.info_loss.train()

        total_loss = 0.0
        total_info = 0.0
        n_batches  = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            images = images.to(self.device)
            B = images.shape[0]

            # One backbone forward populates both _trajectory and _flow_targets
            trajectory   = self.backbone(images)
            flow_targets = self.backbone._flow_targets   # list of L x [B, D_flow]
            L = len(trajectory)

            # Meta-encoder forward
            z_list = self.meta_encoder(trajectory)   # list of L x [B, d]

            # All unique pairs
            idx_a, idx_b = torch.triu_indices(B, B, offset=1, device=self.device)

            z_pairs_a = [z_l[idx_a] for z_l in z_list]
            z_pairs_b = [z_l[idx_b] for z_l in z_list]

            # --- L_info --- flow co-activation targets: f_l^a ⊙ f_l^b
            flow_coact = [
                flow_targets[l][idx_a] * flow_targets[l][idx_b]
                for l in range(L)
            ]   # list of L x [N_pairs, D_flow]
            info_loss = self.info_loss(z_pairs_a, z_pairs_b, flow_coact)

            loss = self.info_loss_weight * info_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.backbone.parameters() if p.requires_grad]
                + list(self.meta_encoder.parameters())
                + list(self.info_loss.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_info += info_loss.item()
            n_batches  += 1

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"info={info_loss.item():.2e}"
                )

        return {
            "loss":      total_loss / max(n_batches, 1),
            "info_loss": total_info / max(n_batches, 1),
        }

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.meta_encoder.eval()
        self.info_loss.eval()

        L = len(self.backbone.layer_dims)
        layer_pred:          list[list] = [[] for _ in range(L)]
        layer_true:          list[list] = [[] for _ in range(L)]
        per_layer_z_sims:    list[list] = [[] for _ in range(L)]
        per_layer_flow_sims: list[list] = [[] for _ in range(L)]

        for images, labels in self.val_loader:
            images = images.to(self.device)
            B = images.shape[0]
            if B < 2:
                continue

            trajectory   = self.backbone(images)
            flow_targets = self.backbone._flow_targets   # list of L x [B, D_flow]
            z_list       = self.meta_encoder(trajectory)

            idx_a, idx_b = torch.triu_indices(B, B, offset=1, device=self.device)
            z_pairs_a = [z_l[idx_a] for z_l in z_list]
            z_pairs_b = [z_l[idx_b] for z_l in z_list]

            # Criterion 1: flow co-activation reconstruction R²
            for l in range(L):
                z_product = z_pairs_a[l] * z_pairs_b[l]
                pred_l = self.info_loss.regressors[l](z_product).cpu()   # [N, D_flow]
                true_l = (flow_targets[l][idx_a] * flow_targets[l][idx_b]).cpu()
                layer_pred[l].append(pred_l)
                layer_true[l].append(true_l)

            # Criterion 2: geometric consistency (Spearman rho)
            for l in range(L):
                z_sim_l    = (z_list[l] @ z_list[l].t())
                flow_sim_l = (flow_targets[l] @ flow_targets[l].t())
                per_layer_z_sims[l].append(z_sim_l[idx_a, idx_b].cpu())
                per_layer_flow_sims[l].append(flow_sim_l[idx_a, idx_b].cpu())

        # Per-layer R²
        per_layer_r2 = []
        for l in range(L):
            pred = torch.cat(layer_pred[l], dim=0).numpy()
            true = torch.cat(layer_true[l], dim=0).numpy()
            ss_res = ((pred - true) ** 2).sum()
            ss_tot = ((true - true.mean()) ** 2).sum()
            per_layer_r2.append(float(1.0 - ss_res / max(ss_tot, 1e-8)))
        r2 = float(np.mean(per_layer_r2))

        # Per-layer Spearman rho
        per_layer_rho = []
        for l in range(L):
            z_all    = torch.cat(per_layer_z_sims[l]).numpy()
            flow_all = torch.cat(per_layer_flow_sims[l]).numpy()
            rho, _   = spearmanr(z_all, flow_all)
            per_layer_rho.append(float(rho) if not np.isnan(rho) else 0.0)

        mean_rho = float(np.mean(per_layer_rho)) if per_layer_rho else 0.0

        return {
            "r2":            r2,
            "mean_rho":      mean_rho,
            "per_layer_rho": per_layer_rho,
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, val_metrics: dict, name: str):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch":               epoch,
                "val_metrics":         val_metrics,
                "backbone_state":      self.backbone.state_dict(),
                "meta_encoder_state":  self.meta_encoder.state_dict(),
                "info_loss_state":     self.info_loss.state_dict(),
                "optimizer_state":     self.optimizer.state_dict(),
                "config":              self.cfg,
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "backbone_state" in ckpt:
            self.backbone.load_state_dict(ckpt["backbone_state"])
        self.meta_encoder.load_state_dict(ckpt["meta_encoder_state"])
        self.info_loss.load_state_dict(ckpt["info_loss_state"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError:
            print("Warning: optimizer state could not be restored; continuing with a fresh optimizer.")
        metrics = ckpt.get("val_metrics", {})
        self._loaded_val_metrics = metrics
        print(
            f"Resumed from {path} (epoch {ckpt['epoch']}, "
            f"R2={metrics.get('r2', 'N/A')}, rho={metrics.get('mean_rho', 'N/A')})"
        )
        return ckpt["epoch"] + 1
