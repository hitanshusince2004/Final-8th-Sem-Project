"""
Glacier Melting Detection — Deep Learning Training Engine
=========================================================
Universal trainer for CNN, U-Net, and DeepLabv3+.

Features:
  - Mixed precision (AMP) training
  - Multiple LR schedulers (cosine, plateau, polynomial)
  - Early stopping with patience
  - TensorBoard-style CSV logging
  - Checkpoint saving (best val mIoU + last)
  - Gradient clipping
  - Warmup epochs
"""

import os
import csv
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
)

from utils.metrics import SegmentationMetrics, print_metrics, save_metrics
from utils.losses  import get_loss
from config.ml_config import CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, SEED


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, config, n_epochs):
    name = config.get("lr_scheduler", "cosine")

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=config.get("patience", 5), min_lr=1e-7,
        )

    elif name == "poly":
        power = config.get("power", 0.9)
        def poly_lr(epoch):
            return max((1 - epoch / n_epochs) ** power, 1e-7)
        return LambdaLR(optimizer, lr_lambda=poly_lr)

    elif name == "warmup_cosine":
        warmup = config.get("warmup_epochs", 5)
        def warmup_cosine(epoch):
            if epoch < warmup:
                return epoch / warmup
            progress = (epoch - warmup) / (n_epochs - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return LambdaLR(optimizer, lr_lambda=warmup_cosine)

    else:
        return CosineAnnealingLR(optimizer, T_max=n_epochs)


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class CSVLogger:
    def __init__(self, path):
        self.path    = path
        self.headers = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, row: dict):
        if self.headers is None:
            self.headers = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.headers).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.headers).writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class GlacierSegTrainer:
    """
    Universal trainer for glacier segmentation models.

    Args:
        model:       nn.Module (GlacierFCN / UNet / DeepLabV3Plus)
        train_loader: DataLoader
        val_loader:   DataLoader
        config:      training hyperparameter dict (from ml_config)
        model_name:  string identifier for checkpoints/logs
    """

    def __init__(self, model, train_loader, val_loader,
                 config, model_name="model"):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.model_name   = model_name.replace(" ", "_").lower()

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n  Device: {self.device}")
        self.model   = self.model.to(self.device)

        # Loss
        self.criterion = get_loss(config.get("loss", "dice_bce"), config)

        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr           = config.get("lr", 3e-4),
            weight_decay = config.get("weight_decay", 1e-4),
        )

        n_epochs = config.get("epochs", 50)
        self.scheduler = build_scheduler(self.optimizer, config, n_epochs)

        # AMP scaler
        self.scaler = GradScaler(enabled=config.get("mixed_precision", True)
                                          and self.device.type == "cuda")

        # Metrics accumulator
        from config.ml_config import N_CLASSES
        self.train_metrics = SegmentationMetrics(num_classes=N_CLASSES)
        self.val_metrics   = SegmentationMetrics(num_classes=N_CLASSES)

        # State
        self.best_miou    = 0.0
        self.epochs_no_improve = 0
        self.history      = []

        # Paths
        self.ckpt_best  = os.path.join(CHECKPOINTS_DIR, f"{self.model_name}_best.pth")
        self.ckpt_last  = os.path.join(CHECKPOINTS_DIR, f"{self.model_name}_last.pth")
        self.log_path   = os.path.join(LOGS_DIR,        f"{self.model_name}_log.csv")
        self.logger     = CSVLogger(self.log_path)

    # ── One training epoch ────────────────────────────────────────────────────

    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        n_batches  = len(self.train_loader)
        
        # Limit training batches on CPU to ensure completion
        max_train_batches = self.config.get("max_train_batches", 500)

        for i, (images, labels) in enumerate(self.train_loader):
            if i >= max_train_batches:
                break
                
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Handle classification vs segmentation labels
            # If labels are (B, H, W) but model output is (B, C), we need (B,) labels
            if labels.ndim == 3 and self.model.__class__.__name__ in ["GlacierCNN", "GlacierResNet"]:
                # Take majority class as patch label
                labels = torch.mode(labels.view(labels.size(0), -1), dim=1)[0]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.train_metrics.update(logits.detach(), labels.detach())

            if (i + 1) % max(1, n_batches // 5) == 0:
                print(f"    Epoch {epoch:3d} [{i+1:4d}/{n_batches}]"
                      f"  loss={loss.item():.4f}")

        return total_loss / n_batches, self.train_metrics.compute()

    # ── Validation epoch ──────────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self):
        import gc
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        gc.collect()

        # Limit validation batches to save time and memory on CPU
        max_val_batches = self.config.get("max_val_batches", 50)
        
        for i, (images, labels) in enumerate(self.val_loader):
            if i >= max_val_batches:
                break
                
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if labels.ndim == 3 and self.model.__class__.__name__ in ["GlacierCNN", "GlacierResNet"]:
                labels = torch.mode(labels.view(labels.size(0), -1), dim=1)[0]

            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item()
            self.val_metrics.update(logits.detach(), labels.detach())
            
            # Explicitly delete to free memory
            del images, labels, logits, loss

        return total_loss / min(len(self.val_loader), max_val_batches), self.val_metrics.compute()

    # ── Full training loop ────────────────────────────────────────────────────

    def train(self):
        n_epochs    = self.config.get("epochs", 50)
        early_stop  = self.config.get("early_stop", 15)
        sched_name  = self.config.get("lr_scheduler", "cosine")

        print(f"\n{'═'*60}")
        print(f"  Training: {self.model_name.upper()}")
        print(f"  Epochs: {n_epochs}  |  Device: {self.device}")
        print(f"  Loss: {self.config.get('loss','?')}  |  LR: {self.config.get('lr')}")
        print(f"{'═'*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # Train
            train_loss, train_m = self.train_epoch(epoch)

            # Validate
            val_loss, val_m = self.val_epoch()

            # LR step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if sched_name == "plateau":
                self.scheduler.step(val_m["mIoU"])
            else:
                self.scheduler.step()

            elapsed = time.time() - t0

            # Log
            log_row = {
                "epoch":       epoch,
                "train_loss":  round(train_loss, 5),
                "val_loss":    round(val_loss, 5),
                "train_mIoU":  train_m["mIoU"],
                "val_mIoU":    val_m["mIoU"],
                "train_F1":    train_m["F1"],
                "val_F1":      val_m["F1"],
                "val_Dice":    val_m["mDice"],
                "val_Accuracy":val_m["Accuracy"],
                "lr":          round(current_lr, 8),
                "elapsed_s":   round(elapsed, 1),
            }
            self.logger.log(log_row)
            self.history.append(log_row)

            print(f"\n  Epoch {epoch:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_mIoU={val_m['mIoU']:.4f}  val_F1={val_m['F1']:.4f}  "
                  f"lr={current_lr:.2e}  [{elapsed:.1f}s]")

            # Save best checkpoint
            if val_m["mIoU"] > self.best_miou:
                self.best_miou = val_m["mIoU"]
                self.epochs_no_improve = 0
                self._save_checkpoint(self.ckpt_best, epoch, val_m)
                print(f"  New best mIoU: {self.best_miou:.4f} -> saved to {self.ckpt_best}")
            else:
                self.epochs_no_improve += 1

            # Save last checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(self.ckpt_last, epoch, val_m)

            # Early stopping
            if self.epochs_no_improve >= early_stop:
                print(f"\n  Early stopping triggered after {early_stop} epochs without improvement.")
                break

        print(f"\n{'═'*60}")
        print(f"  Training complete. Best val mIoU = {self.best_miou:.4f}")
        print(f"  Best checkpoint: {self.ckpt_best}")
        print(f"  Log: {self.log_path}")
        print(f"{'═'*60}")

        return self.history

    # ── Test evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_test(self, test_loader, load_best=True):
        """Load best checkpoint and evaluate on test set."""
        if load_best and os.path.exists(self.ckpt_best):
            ckpt = torch.load(self.ckpt_best, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded best checkpoint (epoch {ckpt['epoch']})")

        self.model.eval()
        test_metrics = SegmentationMetrics(num_classes=self.val_metrics.num_classes)
        total_loss   = 0.0

        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if labels.ndim == 3 and self.model.__class__.__name__ in ["GlacierCNN", "GlacierResNet"]:
                labels = torch.mode(labels.view(labels.size(0), -1), dim=1)[0]
                
            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)
            total_loss += loss.item()
            test_metrics.update(logits, labels)

        result = test_metrics.compute()
        result["test_loss"] = round(total_loss / len(test_loader), 5)
        result["model"]     = self.model_name

        print_metrics(result, model_name=f"{self.model_name} [TEST]")
        save_metrics(result, os.path.join(RESULTS_DIR, f"{self.model_name}_test_metrics.json"))
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def plot_comprehensive_results(self, test_loader, save_dir=None):
        """Generate comprehensive plots for the deep learning model."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        import seaborn as sns
        import gc

        save_dir = save_dir or RESULTS_DIR
        os.makedirs(save_dir, exist_ok=True)
        base_path = os.path.join(save_dir, f"{self.model_name}")

        # 1. Training Curves
        self.plot_training_curves(save_path=f"{base_path}_training_curves.png")

        # 2. Confusion Matrix & Sample Visualizations
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # Get a few samples for visualization
        samples_to_viz = 3
        viz_count = 0
        
        # Subsample for confusion matrix to avoid memory OOM (max 2M pixels total)
        max_total_pixels = 2_000_000
        total_pixels_collected = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                is_clf = self.model.__class__.__name__ in ["GlacierCNN", "GlacierResNet"]
                
                logits = self.model(images)
                preds  = torch.argmax(logits, dim=1).detach().cpu()
                labels_cpu = labels.detach().cpu()
                
                if is_clf:
                    labels_clf = torch.mode(labels_cpu.view(labels_cpu.size(0), -1), dim=1)[0]
                    all_preds.append(preds.numpy().astype(np.uint8))
                    all_labels.append(labels_clf.numpy().astype(np.uint8))
                else:
                    # Subsample pixels for segmentation to avoid OOM
                    # Keep only every Nth pixel if needed
                    B, H, W = labels_cpu.shape
                    batch_pixels = B * H * W
                    
                    if total_pixels_collected < max_total_pixels:
                        # For segmentation, we take a subset of batches or a subset of pixels
                        # Let's just take the first few batches until we hit the limit
                        p = preds.numpy().astype(np.uint8).flatten()
                        l = labels_cpu.numpy().astype(np.uint8).flatten()
                        all_preds.append(p)
                        all_labels.append(l)
                        total_pixels_collected += batch_pixels
                
                if viz_count < samples_to_viz:
                    for i in range(min(images.size(0), samples_to_viz - viz_count)):
                        if not is_clf:
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                            axes[0].imshow(labels_cpu[i].numpy(), cmap="tab10")
                            axes[0].set_title(f"Actual Mask (S{viz_count+1})")
                            axes[1].imshow(preds[i].numpy(), cmap="tab10")
                            axes[1].set_title(f"Predicted Mask (S{viz_count+1})")
                            plt.tight_layout()
                            plt.savefig(f"{base_path}_sample_{viz_count+1}_pred_vs_actual.png")
                            plt.close()
                            gc.collect()
                        viz_count += 1
                
                del images, labels_cpu, logits, preds
                gc.collect()
                
        if not all_preds: return # Avoid error if empty

        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        
        # 3. Confusion Matrix
        classes = ["Land", "Snow/Ice", "Water", "Debris"]
        present_classes = np.unique(np.concatenate([all_labels, all_preds])).astype(int)
        display_labels = [classes[i] for i in present_classes]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            all_labels, all_preds, labels=present_classes, 
            display_labels=display_labels,
            cmap="Blues", normalize="true", ax=ax
        )
        ax.set_title(f"{self.model_name} — Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{base_path}_confusion_matrix.png")
        plt.close()
        gc.collect()

        # 4. Class Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        unique_act, counts_act = np.unique(all_labels, return_counts=True)
        unique_pred, counts_pred = np.unique(all_preds, return_counts=True)
        
        # Ensure all classes are represented
        act_dist = np.zeros(len(classes))
        pred_dist = np.zeros(len(classes))
        for u, c in zip(unique_act, counts_act): act_dist[int(u)] = c
        for u, c in zip(unique_pred, counts_pred): pred_dist[int(u)] = c
        
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, act_dist, width, label="Actual", color="blue", alpha=0.6)
        ax.bar(x + width/2, pred_dist, width, label="Predicted", color="orange", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_title(f"{self.model_name} — Class Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{base_path}_distribution.png")
        plt.close()
        gc.collect()

        print(f"  ✓ Comprehensive plots saved to {save_dir}")

    def _save_checkpoint(self, path, epoch, metrics):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "metrics":     metrics,
            "best_miou":   self.best_miou,
            "config":      self.config,
        }, path)

    def plot_training_curves(self, save_path=None):
        """Plot loss and mIoU training/validation curves."""
        import matplotlib.pyplot as plt

        if not self.history:
            print("  No training history to plot.")
            return

        epochs     = [h["epoch"]      for h in self.history]
        train_loss = [h["train_loss"] for h in self.history]
        val_loss   = [h["val_loss"]   for h in self.history]
        train_miou = [h["train_mIoU"] for h in self.history]
        val_miou   = [h["val_mIoU"]   for h in self.history]
        val_f1     = [h["val_F1"]     for h in self.history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss curve
        axes[0].plot(epochs, train_loss, label="Train", color="#3B8BD4")
        axes[0].plot(epochs, val_loss,   label="Val",   color="#E8593C")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{self.model_name} — Loss curve")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # mIoU curve
        axes[1].plot(epochs, train_miou, label="Train mIoU", color="#3B8BD4")
        axes[1].plot(epochs, val_miou,   label="Val mIoU",   color="#1D9E75")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
        axes[1].set_title(f"{self.model_name} — mIoU curve")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        # F1 curve
        axes[2].plot(epochs, val_f1, label="Val F1", color="#534AB7")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1")
        axes[2].set_title(f"{self.model_name} — Val F1 curve")
        axes[2].legend(); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Training curves saved → {save_path}")
        plt.close()
        return fig
