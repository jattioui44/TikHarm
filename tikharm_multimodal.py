"""
TikHarm — Multimodal Audio+Video Classifier
Combines PE Core (timm ViT) for video and Wav2Vec2 for audio.

Fusion methods (cfg.fusion_method):
    concat          : concatenation [video_emb ; audio_emb]
    gated           : sigmoid gate fusion
    cross_attention : bidirectional multi-head cross-attention
    transformer     : TransformerEncoder on [CLS, video, audio]
    weighted        : learned softmax-weighted sum

Requirements:
    pip install timm torch torchvision torchaudio opencv-python-headless
    pip install scikit-learn pandas Pillow transformers av
"""

import os
import json
import math
import warnings
import pathlib

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

os.environ["WANDB_DISABLED"]         = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

class Config:
    # Dataset root: expects train/val/test subfolders, each with class subfolders
    dataset_root     = "path/to/tikharm"
    model_output_dir = "./pe-video-wav2vec2-multimodal"

    # Backbones
    timm_model_name  = "vit_pe_core_base_patch16_224.fb"
    audio_model_name = "facebook/wav2vec2-base"

    num_classes = 4  # updated automatically from folder names

    # Video
    max_frames = 8

    # Audio
    sample_rate  = 16000
    max_duration = 10.0
    max_length   = 160000   # sample_rate * max_duration

    # Training
    num_epochs       = 10
    train_batch_size = 4
    eval_batch_size  = 4
    learning_rate    = 1e-5
    weight_decay     = 0.01
    warmup_ratio     = 0.1
    label_smoothing  = 0.1

    # Freeze backbones (set True to train only the fusion head)
    freeze_video_backbone = False
    freeze_audio_backbone = False

    # Fusion method: concat | gated | cross_attention | transformer | weighted
    fusion_method = "concat"

    # Shared fusion hyperparameters
    fusion_proj_dim   = 512
    fusion_num_heads  = 8
    fusion_num_layers = 2
    fusion_dropout    = 0.1


# ---------------------------------------------------------------------------
# DATASET LOADER
# ---------------------------------------------------------------------------

def load_dataset_from_folders(cfg):
    """Scans dataset_root/split/class/*.mp4 and returns a DataFrame."""
    root    = pathlib.Path(cfg.dataset_root)
    records = []

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            print("[WARN] Folder not found: {}".format(split_dir))
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for video_path in class_dir.glob("*.mp4"):
                records.append({
                    "video_path": str(video_path),
                    "class_name": class_dir.name,
                    "split":      split,
                })

    df = pd.DataFrame(records)
    print("\n[INFO] Dataset loaded from: {}".format(cfg.dataset_root))
    print("[INFO] Total videos: {}".format(len(df)))
    print(df.groupby(["split", "class_name"]).size().to_string())
    return df


# ---------------------------------------------------------------------------
# VIDEO LOADER
# ---------------------------------------------------------------------------

class VideoLoader:
    """Extracts max_frames evenly spaced frames from a video file."""

    def __init__(self, cfg, image_transform):
        self.cfg       = cfg
        self.transform = image_transform

    def load(self, video_path):
        try:
            cap   = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or total <= 0:
                cap.release()
                return self._dummy()

            indices = np.linspace(0, total - 1, self.cfg.max_frames, dtype=int)
            frames  = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
                else:
                    frames.append(torch.zeros(3, 224, 224))
            cap.release()

            while len(frames) < self.cfg.max_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))

            return torch.stack(frames[:self.cfg.max_frames])

        except Exception:
            return self._dummy()

    def _dummy(self):
        return torch.zeros(self.cfg.max_frames, 3, 224, 224)


# ---------------------------------------------------------------------------
# AUDIO EXTRACTOR
# ---------------------------------------------------------------------------

class AudioExtractor:
    """Extracts and resamples audio from a video file using PyAV."""

    def __init__(self, cfg):
        self.sample_rate = cfg.sample_rate
        self.max_samples = int(cfg.sample_rate * cfg.max_duration)
        self.success = 0
        self.failed  = 0

    def extract(self, video_path):
        try:
            import av
            container = av.open(video_path)
            if not container.streams.audio:
                container.close()
                return self._silence()

            sr_original = container.streams.audio[0].sample_rate
            samples = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                samples.append(arr.mean(axis=0) if arr.ndim == 2 else arr)
            container.close()

            if not samples:
                return self._silence()

            audio = np.concatenate(samples).astype(np.float32)

            if sr_original != self.sample_rate:
                import torchaudio.transforms as T
                resampler = T.Resample(orig_freq=sr_original, new_freq=self.sample_rate)
                audio     = resampler(torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()

            audio = audio[:self.max_samples]
            if np.abs(audio).max() < 1e-6:
                return self._silence()

            audio = audio / np.abs(audio).max()
            self.success += 1
            return audio.astype(np.float32)

        except Exception as e:
            self.failed += 1
            if self.failed <= 5:
                print("[WARN] Audio error ({}): {}".format(self.failed, e))
            return self._silence()

    def _silence(self):
        self.failed += 1
        return np.zeros(self.max_samples, dtype=np.float32)

    def print_stats(self):
        total = self.success + self.failed
        if total > 0:
            print("[INFO] Audio loaded: {}/{} ({:.1f}%) | Failed: {}".format(
                self.success, total, 100 * self.success / total, self.failed
            ))


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """Returns video frames, audio input values and class label for each sample."""

    def __init__(self, df, class_to_id, cfg, image_transform, feature_extractor):
        self.df                = df.reset_index(drop=True)
        self.class_to_id       = class_to_id
        self.video_loader      = VideoLoader(cfg, image_transform)
        self.audio_extractor   = AudioExtractor(cfg)
        self.feature_extractor = feature_extractor
        self.cfg               = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        video_path = str(row["video_path"])

        video_tensor = self.video_loader.load(video_path)  # (T, C, H, W)

        audio_raw    = self.audio_extractor.extract(video_path)
        audio_inputs = self.feature_extractor(
            audio_raw,
            sampling_rate=self.cfg.sample_rate,
            max_length=self.cfg.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "pixel_values": video_tensor,
            "input_values": audio_inputs["input_values"].squeeze(0),
            "labels":       torch.tensor(self.class_to_id[row["class_name"]], dtype=torch.long),
        }

    def print_stats(self):
        self.audio_extractor.print_stats()


# ---------------------------------------------------------------------------
# FUSION MODULES
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """Simple concatenation: [video || audio] → LayerNorm → Linear → GELU → Linear."""

    def __init__(self, video_dim, audio_dim, proj_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(video_dim + audio_dim),
            nn.Linear(video_dim + audio_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, video_emb, audio_emb):
        return self.net(torch.cat([video_emb, audio_emb], dim=1))


class GatedFusion(nn.Module):
    """
    Sigmoid gate fusion.
    g = Sigmoid(Linear([v; a]))
    fused = g * proj(v) + (1-g) * proj(a)
    """

    def __init__(self, video_dim, audio_dim, proj_dim, num_classes, dropout):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(video_dim + audio_dim, proj_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, video_emb, audio_emb):
        v = self.video_proj(video_emb)
        a = self.audio_proj(audio_emb)
        g = self.gate(torch.cat([video_emb, audio_emb], dim=1))
        return self.classifier(g * v + (1.0 - g) * a)


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention.
    v_ctx = Attn(Q=v, K=a, V=a)
    a_ctx = Attn(Q=a, K=v, V=v)
    fused = v_ctx + a_ctx
    """

    def __init__(self, video_dim, audio_dim, proj_dim, num_heads, num_classes, dropout):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.v2a_attn   = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.a2v_attn   = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, video_emb, audio_emb):
        v = self.video_proj(video_emb).unsqueeze(1)   # (B, 1, proj)
        a = self.audio_proj(audio_emb).unsqueeze(1)
        v_ctx, _ = self.v2a_attn(query=v, key=a, value=a)
        a_ctx, _ = self.a2v_attn(query=a, key=v, value=v)
        return self.classifier((v_ctx + a_ctx).squeeze(1))


class TransformerFusion(nn.Module):
    """
    TransformerEncoder on sequence [CLS, video_proj, audio_proj].
    Uses the CLS token output for classification.
    """

    def __init__(self, video_dim, audio_dim, proj_dim, num_heads, num_layers, num_classes, dropout):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, 2, proj_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=num_heads,
            dim_feedforward=proj_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, video_emb, audio_emb):
        B  = video_emb.size(0)
        v  = self.video_proj(video_emb).unsqueeze(1)
        a  = self.audio_proj(audio_emb).unsqueeze(1)
        seq = torch.cat([self.cls_token.expand(B, -1, -1),
                          torch.cat([v, a], dim=1) + self.pos_embed], dim=1)
        return self.classifier(self.transformer(seq)[:, 0, :])


class WeightedFusion(nn.Module):
    """
    Learned softmax-weighted sum of projected modalities.
    fused = softmax([w_v, w_a])[0] * proj(v) + softmax(...)[1] * proj(a)
    """

    def __init__(self, video_dim, audio_dim, proj_dim, num_classes, dropout):
        super().__init__()
        self.video_proj    = nn.Linear(video_dim, proj_dim)
        self.audio_proj    = nn.Linear(audio_dim, proj_dim)
        self.modal_weights = nn.Parameter(torch.ones(2))
        self.classifier    = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, video_emb, audio_emb):
        w = F.softmax(self.modal_weights, dim=0)
        return self.classifier(w[0] * self.video_proj(video_emb) +
                                w[1] * self.audio_proj(audio_emb))

    def get_weights(self):
        with torch.no_grad():
            w = F.softmax(self.modal_weights, dim=0).cpu().numpy()
        return {"video": float(w[0]), "audio": float(w[1])}


# ---------------------------------------------------------------------------
# MULTIMODAL CLASSIFIER
# ---------------------------------------------------------------------------

class MultimodalClassifier(nn.Module):
    """
    Video branch  : PE Core ViT  → mean-pool over frames → video_emb
    Audio branch  : Wav2Vec2     → mean-pool over time   → audio_emb
    Fusion module : selected by cfg.fusion_method
    """

    FUSION_METHODS = ("concat", "gated", "cross_attention", "transformer", "weighted")

    def __init__(self, cfg):
        super().__init__()

        if cfg.fusion_method not in self.FUSION_METHODS:
            raise ValueError("Unknown fusion_method '{}'. Choose from: {}".format(
                cfg.fusion_method, self.FUSION_METHODS))

        self.fusion_method   = cfg.fusion_method
        self.label_smoothing = cfg.label_smoothing

        # Video backbone
        print("[INFO] Loading video backbone: {}".format(cfg.timm_model_name))
        self.video_backbone  = timm.create_model(cfg.timm_model_name, pretrained=True, num_classes=0)
        self.video_embed_dim = self.video_backbone.num_features
        if cfg.freeze_video_backbone:
            for p in self.video_backbone.parameters():
                p.requires_grad = False

        # Audio backbone
        print("[INFO] Loading audio backbone: {}".format(cfg.audio_model_name))
        self.audio_backbone  = Wav2Vec2Model.from_pretrained(cfg.audio_model_name)
        self.audio_embed_dim = self.audio_backbone.config.hidden_size
        if cfg.freeze_audio_backbone:
            for p in self.audio_backbone.parameters():
                p.requires_grad = False

        V, A, P, H, L, C, D = (
            self.video_embed_dim, self.audio_embed_dim,
            cfg.fusion_proj_dim, cfg.fusion_num_heads,
            cfg.fusion_num_layers, cfg.num_classes, cfg.fusion_dropout,
        )

        fusion_map = {
            "concat":          lambda: ConcatFusion(V, A, P, C, D),
            "gated":           lambda: GatedFusion(V, A, P, C, D),
            "cross_attention": lambda: CrossAttentionFusion(V, A, P, H, C, D),
            "transformer":     lambda: TransformerFusion(V, A, P, H, L, C, D),
            "weighted":        lambda: WeightedFusion(V, A, P, C, D),
        }
        self.fusion = fusion_map[cfg.fusion_method]()
        print("[INFO] Fusion: {}  (proj_dim={})".format(cfg.fusion_method, P))

    def encode_video(self, pixel_values):
        """(B, T, C, H, W) → (B, V)"""
        B, T, C, H, W = pixel_values.shape
        emb = self.video_backbone(pixel_values.reshape(B * T, C, H, W))
        return emb.reshape(B, T, -1).mean(dim=1)

    def encode_audio(self, input_values):
        """(B, max_length) → (B, A)"""
        return self.audio_backbone(input_values=input_values).last_hidden_state.mean(dim=1)

    def forward(self, pixel_values, input_values, labels=None, **kwargs):
        logits = self.fusion(self.encode_video(pixel_values), self.encode_audio(input_values))
        loss   = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(logits, labels)
        return {"loss": loss, "logits": logits}

    def log_fusion_weights(self):
        if self.fusion_method == "weighted":
            w = self.fusion.get_weights()
            print("[INFO] Fusion weights — video: {:.4f} | audio: {:.4f}".format(
                w["video"], w["audio"]))


# ---------------------------------------------------------------------------
# COLLATE & METRICS
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_score":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    model.train()
    total_loss, num_batches = 0, 0

    for step, batch in enumerate(dataloader):
        pv = batch["pixel_values"].to(device)
        iv = batch["input_values"].to(device)
        lb = batch["labels"].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model(pixel_values=pv, input_values=iv, labels=lb)["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(pixel_values=pv, input_values=iv, labels=lb)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

        total_loss  += loss.item()
        num_batches += 1

        if (step + 1) % 50 == 0:
            print("  [Epoch {}] Step {}/{} - Loss: {:.4f}".format(
                epoch + 1, step + 1, len(dataloader), total_loss / num_batches))

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, total_loss, num_batches = [], [], 0, 0

    for batch in dataloader:
        pv = batch["pixel_values"].to(device)
        iv = batch["input_values"].to(device)
        lb = batch["labels"].to(device)

        outputs = model(pixel_values=pv, input_values=iv, labels=lb)
        all_preds.extend(outputs["logits"].argmax(dim=-1).cpu().numpy())
        all_labels.extend(lb.cpu().numpy())
        total_loss  += outputs["loss"].item()
        num_batches += 1

    metrics       = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics, np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes, fusion_method, output_dir):
    """Saves a normalized confusion matrix as PNG."""
    cm      = confusion_matrix(y_true, y_pred)
    n       = len(classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    cmap = LinearSegmentedColormap.from_list(
        "tikharm", ["#FFFFFF", "#C8C3F4", "#7F77DD", "#3C3489"])

    fig, ax = plt.subplots(figsize=(max(6, n * 1.6), max(5, n * 1.4)))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall per class", fontsize=10)

    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion matrix — test set\n(fusion: {})".format(fusion_method), fontsize=12)

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.55 else "#2C2C2A"
            ax.text(j, i - 0.08, str(cm[i, j]),
                    ha="center", va="center", fontsize=13, fontweight="bold", color=color)
            ax.text(j, i + 0.22, "{:.1f}%".format(cm_norm[i, j] * 100),
                    ha="center", va="center", fontsize=8, color=color, alpha=0.85)

    ax.set_xticks(ticks - 0.5, minor=True); ax.set_yticks(ticks - 0.5, minor=True)
    ax.grid(which="minor", color="#D3D1C7", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_{}.png".format(fusion_method))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[PLOT] Confusion matrix saved → {}".format(path))
    return path


def plot_learning_curves(history, fusion_method, output_dir):
    """Saves train/val loss and val accuracy/F1 curves as PNG."""
    epochs  = list(range(1, len(history["train_loss"]) + 1))
    palette = {"train_loss": "#D85A30", "val_loss": "#7F77DD",
                "val_acc": "#1D9E75", "val_f1": "#BA7517"}

    fig, (ax_loss, ax_score) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                             gridspec_kw={"hspace": 0.08})
    fig.suptitle("Learning curves — fusion: {}".format(fusion_method), fontsize=13)

    ax_loss.plot(epochs, history["train_loss"], color=palette["train_loss"], lw=2,
                  marker="o", ms=5, label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], color=palette["val_loss"], lw=2,
                  marker="s", ms=5, linestyle="--", label="Val loss")
    ax_loss.set_ylabel("Loss"); ax_loss.legend(fontsize=9); ax_loss.grid(alpha=0.4)
    ax_loss.spines[["top", "right"]].set_visible(False)

    ax_score.plot(epochs, history["val_acc"], color=palette["val_acc"], lw=2,
                   marker="o", ms=5, label="Val accuracy")
    ax_score.plot(epochs, history["val_f1"], color=palette["val_f1"], lw=2,
                   marker="D", ms=5, linestyle="--", label="Val F1 (macro)")
    ax_score.set_xlabel("Epoch"); ax_score.set_ylabel("Score")
    ax_score.set_ylim(0, 1.05)
    ax_score.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_score.legend(fontsize=9); ax_score.grid(alpha=0.4)
    ax_score.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_score.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "learning_curves_{}.png".format(fusion_method))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[PLOT] Learning curves saved → {}".format(path))
    return path


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(cfg):
    df = load_dataset_from_folders(cfg)
    if len(df) == 0:
        raise RuntimeError("No videos found in {}".format(cfg.dataset_root))

    classes         = sorted(df["class_name"].unique())
    class_to_id     = {c: i for i, c in enumerate(classes)}
    cfg.num_classes = len(classes)
    print("Classes ({}): {}".format(cfg.num_classes, classes))

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]
    print("Train: {} | Val: {} | Test: {}".format(len(train_df), len(val_df), len(test_df)))

    # Transforms
    data_cfg        = resolve_data_config(model=timm.create_model(cfg.timm_model_name, pretrained=False))
    image_transform = create_transform(**data_cfg, is_training=False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.audio_model_name)

    # Datasets
    train_ds = MultimodalDataset(train_df, class_to_id, cfg, image_transform, feature_extractor)
    val_ds   = MultimodalDataset(val_df,   class_to_id, cfg, image_transform, feature_extractor)
    test_ds  = MultimodalDataset(test_df,  class_to_id, cfg, image_transform, feature_extractor)

    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size,
                               shuffle=True,  num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size,
                               shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.eval_batch_size,
                               shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Sanity check on first sample
    sample = train_ds[0]
    print("[TEST] pixel_values: {} | input_values: {} | label: {}".format(
        sample["pixel_values"].shape, sample["input_values"].shape, sample["labels"].item()))
    train_ds.print_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    model = MultimodalClassifier(cfg).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: {:,} | Trainable: {:,}".format(total, trainable))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.learning_rate,
        total_steps=len(train_loader) * cfg.num_epochs, pct_start=cfg.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    os.makedirs(cfg.model_output_dir, exist_ok=True)
    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(cfg.num_epochs):
        print("\n" + "=" * 60)
        print("EPOCH {}/{}  [fusion={}]".format(epoch + 1, cfg.num_epochs, cfg.fusion_method))
        print("=" * 60)

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, scaler)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        print("  Train Loss: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f} | Val F1: {:.4f}".format(
            train_loss, val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1_score"]))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])

        model.log_fusion_weights()

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), os.path.join(cfg.model_output_dir, "best_model.pt"))
            print("  → Best model saved (F1={:.4f})".format(best_f1))

    # Test evaluation
    print("\nLoading best model (Val F1={:.4f})...".format(best_f1))
    model.load_state_dict(torch.load(os.path.join(cfg.model_output_dir, "best_model.pt")))
    test_metrics, y_true, y_pred = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS  [fusion={}]".format(cfg.fusion_method))
    print("=" * 60)
    print("  Accuracy : {:.4f}".format(test_metrics["accuracy"]))
    print("  F1-Score : {:.4f}".format(test_metrics["f1_score"]))
    print("  Precision: {:.4f}".format(test_metrics["precision"]))
    print("  Recall   : {:.4f}".format(test_metrics["recall"]))
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    model.log_fusion_weights()

    # Save plots and results
    plot_confusion_matrix(y_true, y_pred, classes, cfg.fusion_method, cfg.model_output_dir)
    plot_learning_curves(history, cfg.fusion_method, cfg.model_output_dir)

    results = {
        "video_model": cfg.timm_model_name, "audio_model": cfg.audio_model_name,
        "fusion": cfg.fusion_method,
        "accuracy": test_metrics["accuracy"], "f1": test_metrics["f1_score"],
        "precision": test_metrics["precision"], "recall": test_metrics["recall"],
        "classes": classes, "history": history,
    }
    out_file = "results_multimodal_{}.json".format(cfg.fusion_method)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved → {}".format(out_file))

    return test_metrics["accuracy"], test_metrics["f1_score"]


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    cfg = Config()
    # Change fusion_method to try other strategies:
    #   concat | gated | cross_attention | transformer | weighted
    cfg.fusion_method = "weighted"

    run(cfg)
