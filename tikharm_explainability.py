"""
TikHarm — Explainability: Grad-CAM (video) + Audio Saliency (Wav2Vec2)
Loads a saved best_model.pt and produces a publication-quality PDF figure:
    (a) Grad-CAM overlays on 4 representative frames
    (b) Gradient x Input saliency on the raw waveform
    (c) Softmax class probability distribution

Usage:
    Edit the CONFIG section at the bottom, then run:
        python tikharm_explainability.py

Additional requirements (beyond tikharm_multimodal.py):
    pip install matplotlib opencv-python-headless
"""

import os
import sys
import warnings
import pathlib

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from transformers import Wav2Vec2FeatureExtractor
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# GRAD-CAM FOR TIMM ViT
# ---------------------------------------------------------------------------

class GradCAMViT:
    """
    Grad-CAM for Vision Transformers.
    Registers forward/backward hooks on the target layer,
    then computes: relu( sum_c( mean(grad_c) * activation_c ) ) on patch tokens.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Returns a (H, W) Grad-CAM map normalized to [0, 1]."""
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        act  = self._activations  # (1, seq_len, dim)
        grad = self._gradients

        if act.dim() == 3:
            # Skip CLS token (index 0), keep patch tokens only
            act_p  = act[:, 1:, :]
            grad_p = grad[:, 1:, :]
            weights    = grad_p.mean(dim=1, keepdim=True)
            cam_tokens = torch.relu((weights * act_p).sum(dim=-1)).squeeze(0).cpu().numpy()
        else:
            cam_tokens = torch.relu(act.squeeze(0)).cpu().numpy()

        # Reshape to square spatial grid
        n    = cam_tokens.shape[0]
        side = int(np.sqrt(n))
        if side * side != n:
            side       = int(np.ceil(np.sqrt(n)))
            cam_tokens = np.pad(cam_tokens, (0, side * side - n))
        cam_2d = cam_tokens.reshape(side, side)

        cam_2d -= cam_2d.min()
        if cam_2d.max() > 1e-8:
            cam_2d /= cam_2d.max()
        return cam_2d


def overlay_gradcam(frame_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends a Grad-CAM heatmap over an RGB frame (uint8)."""
    H, W        = frame_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    return (alpha * heatmap_rgb + (1 - alpha) * frame_rgb).astype(np.uint8)


# ---------------------------------------------------------------------------
# AUDIO SALIENCY
# ---------------------------------------------------------------------------

class AudioSaliency:
    """
    Gradient x Input saliency on the raw waveform fed into Wav2Vec2.
    Enables gradient tracking on input_values, runs backward on the
    predicted class score, then computes |grad * input|.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute(self, input_values, pixel_values, class_idx=None, smooth_window=400):
        """Returns normalized saliency array of shape (T,)."""
        self.model.eval()
        x = input_values.clone().detach().requires_grad_(True)

        outputs = self.model(pixel_values=pixel_values, input_values=x)
        logits  = outputs["logits"]

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        saliency = (x.grad * x).abs().squeeze(0).detach().cpu().numpy()

        if smooth_window > 1 and len(saliency) > smooth_window:
            saliency = np.convolve(saliency, np.ones(smooth_window) / smooth_window, mode="same")

        saliency -= saliency.min()
        if saliency.max() > 1e-8:
            saliency /= saliency.max()
        return saliency


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def explain_video(video_path, model_path, fusion_method, output_dir,
                   num_frames=8, classes=None):
    """
    Full explainability pipeline:
      1. Load checkpoint
      2. Extract frames and audio
      3. Forward pass → prediction + probabilities
      4. Grad-CAM on video branch
      5. Gradient x Input on audio branch
      6. Save summary PDF figure
    """
    from tikharm_multimodal import Config, MultimodalClassifier, VideoLoader, AudioExtractor

    os.makedirs(output_dir, exist_ok=True)

    cfg               = Config()
    cfg.fusion_method = fusion_method
    cfg.max_frames    = num_frames
    if classes:
        cfg.num_classes = len(classes)

    data_cfg          = resolve_data_config(model=timm.create_model(cfg.timm_model_name, pretrained=False))
    image_transform   = create_transform(**data_cfg, is_training=False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.audio_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device: {}".format(device))

    model = MultimodalClassifier(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[INFO] Model loaded: {}".format(model_path))

    # Load video frames
    video_loader = VideoLoader(cfg, image_transform)
    pixel_values = video_loader.load(video_path).unsqueeze(0).to(device)  # (1, T, C, H, W)

    cap        = cv2.VideoCapture(video_path)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_frames = []
    for idx in np.linspace(0, total - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else
                           np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()

    # Load audio
    audio_extractor = AudioExtractor(cfg)
    audio_raw       = audio_extractor.extract(video_path)
    audio_inputs    = feature_extractor(
        audio_raw, sampling_rate=cfg.sample_rate,
        max_length=cfg.max_length, truncation=True,
        padding="max_length", return_tensors="pt")
    input_values = audio_inputs["input_values"].to(device)

    # Prediction
    with torch.no_grad():
        outputs  = model(pixel_values=pixel_values, input_values=input_values)
        probs    = torch.softmax(outputs["logits"], dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())

    pred_label = classes[pred_idx] if classes else "Class {}".format(pred_idx)
    print("[INFO] Prediction: {} ({:.1f}%)".format(pred_label, 100 * probs[pred_idx]))

    # Grad-CAM
    print("[INFO] Computing Grad-CAM...")
    target_layer = _find_vit_target_layer(model.video_backbone)
    grad_cam     = GradCAMViT(model.video_backbone, target_layer)
    cam_frames   = [grad_cam(pixel_values[0, t].unsqueeze(0)) for t in range(num_frames)]
    grad_cam.remove_hooks()

    # Audio saliency
    print("[INFO] Computing audio saliency...")
    saliency = AudioSaliency(model).compute(input_values, pixel_values,
                                             class_idx=pred_idx, smooth_window=800)

    # Generate figure
    print("[INFO] Generating figure...")
    _plot_explainability(
        raw_frames=raw_frames, cam_frames=cam_frames,
        audio_raw=audio_raw, saliency=saliency,
        probs=probs, pred_label=pred_label, classes=classes,
        output_dir=output_dir, video_name=pathlib.Path(video_path).stem)

    return pred_label, probs


def _find_vit_target_layer(vit_backbone: nn.Module) -> nn.Module:
    """Returns the best Grad-CAM target layer for a timm ViT backbone."""
    if hasattr(vit_backbone, "blocks") and len(vit_backbone.blocks) > 0:
        last_block = vit_backbone.blocks[-1]
        return last_block.norm1 if hasattr(last_block, "norm1") else last_block
    if hasattr(vit_backbone, "norm"):
        return vit_backbone.norm
    print("[WARN] Target layer not found automatically, using full backbone.")
    return vit_backbone


def _plot_explainability(raw_frames, cam_frames, audio_raw, saliency,
                          probs, pred_label, classes, output_dir, video_name):
    """
    Saves a 3-panel explainability figure as PDF:
        (a) Grad-CAM on 4 key frames
        (b) Audio saliency (Gradient x Input)
        (c) Class probability bars
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    import matplotlib.ticker as mticker

    PURPLE = "#7F77DD"; CORAL = "#D85A30"; TEAL = "#1D9E75"
    DARK   = "#1A1A1A"; GRAY  = "#888780"; GRAY_LT = "#C8C5BE"

    n_total     = len(raw_frames)
    key_indices = sorted(set([0, n_total // 3, 2 * n_total // 3, n_total - 1]))
    sel_frames  = [raw_frames[i] for i in key_indices]
    sel_cams    = [cam_frames[i] for i in key_indices]
    n_sel       = len(sel_frames)

    fig = plt.figure(figsize=(16, 11), facecolor="#FFFFFF")
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5.2, 3.2], hspace=0.08)

    # --- Panel (a): Grad-CAM frames ---
    ax_frames = gridspec.GridSpecFromSubplotSpec(1, n_sel, subplot_spec=outer[0], wspace=0.04)
    for i, (frame_rgb, cam) in enumerate(zip(sel_frames, sel_cams)):
        ax  = fig.add_subplot(ax_frames[0, i])
        H, W = frame_rgb.shape[:2]
        ax.imshow(overlay_gradcam(frame_rgb, cam, alpha=0.42))

        # Circle marking peak saliency location
        cam_up = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        cy, cx = np.unravel_index(np.argmax(cam_up), cam_up.shape)
        r      = max(12, min(H, W) // 8)
        ax.add_patch(plt.Circle((cx, cy), r, edgecolor="white", facecolor="none", lw=2.0))
        ax.add_patch(plt.Circle((cx, cy), r, edgecolor=CORAL,   facecolor="none", lw=0.8, alpha=0.85))

        t_sec = key_indices[i] / max(n_total - 1, 1) * (len(audio_raw) / 16000)
        ax.set_title("t = {:.1f} s".format(t_sec), fontsize=9, color=DARK, pad=4)
        ax.text(0.03, 0.03, "(a{})".format(i + 1), transform=ax.transAxes,
                fontsize=8, color="white", fontweight="bold", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))
        ax.axis("off")

    # Shared Grad-CAM colorbar
    cbar_ax = fig.add_axes([0.955, 0.53, 0.012, 0.34])
    sm      = ScalarMappable(cmap="jet", norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Grad-CAM\nsaliency", fontsize=7.5, color=DARK, labelpad=4)
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=7)

    # --- Bottom row ---
    bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                            wspace=0.22, width_ratios=[2.2, 1.0])

    # Panel (b): Audio saliency
    ax_audio = fig.add_subplot(bot[0, 0])
    T        = len(audio_raw)
    t_axis   = np.linspace(0, T / 16000, T)
    ax_audio.fill_between(t_axis, audio_raw, -audio_raw, alpha=0.18, color=GRAY)
    ax_audio.plot(t_axis, audio_raw, color=GRAY_LT, lw=0.35, alpha=0.7)

    ax_sal = ax_audio.twinx()
    t_sal  = np.linspace(0, T / 16000, len(saliency))
    ax_sal.plot(t_sal, saliency, color=CORAL, lw=1.6, zorder=3)

    above = saliency > 0.6
    if above.any():
        ax_sal.fill_between(t_sal, 0, saliency, where=above, alpha=0.22, color=CORAL, zorder=2)

    ax_sal.set_ylim(-0.05, 1.25)
    ax_sal.set_ylabel("Normalized saliency", fontsize=8.5, color=CORAL, labelpad=6)
    ax_sal.tick_params(axis="y", labelcolor=CORAL, labelsize=7.5)
    ax_sal.spines["right"].set_edgecolor(CORAL)
    ax_sal.spines["top"].set_visible(False)

    ax_audio.set_xlabel("Time (s)", fontsize=9); ax_audio.set_ylabel("Amplitude", fontsize=8.5, color=GRAY)
    ax_audio.set_title("(b) Audio saliency — Gradient × Input on Wav2Vec2",
                         fontsize=9.5, color=DARK, pad=6, loc="left")
    ax_audio.spines[["top", "right"]].set_visible(False)
    ax_audio.grid(color=GRAY_LT, lw=0.3, linestyle="--", axis="x", alpha=0.6)

    legend_handles = [
        Line2D([0], [0], color=GRAY,  lw=1.2, alpha=0.6, label="Raw waveform"),
        Line2D([0], [0], color=CORAL, lw=1.6,             label="Audio saliency (Grad × Input)"),
    ]
    ax_audio.legend(handles=legend_handles, fontsize=7.5, loc="upper right",
                     framealpha=0.7, fancybox=False)

    # Panel (c): Class probabilities
    ax_prob  = fig.add_subplot(bot[0, 1])
    pred_idx = int(probs.argmax())
    labels_b = classes if classes else ["Class {}".format(i) for i in range(len(probs))]
    colors_b = [TEAL if i == pred_idx else PURPLE for i in range(len(probs))]

    bars = ax_prob.barh(labels_b, probs * 100, color=colors_b, height=0.52)
    for bar, p, c in zip(bars, probs, colors_b):
        w = bar.get_width()
        if w > 15:
            ax_prob.text(w - 1.5, bar.get_y() + bar.get_height() / 2,
                          "{:.1f}%".format(p * 100), va="center", ha="right",
                          fontsize=8, color="white", fontweight="bold")
        else:
            ax_prob.text(w + 1.0, bar.get_y() + bar.get_height() / 2,
                          "{:.1f}%".format(p * 100), va="center", ha="left",
                          fontsize=8, color=DARK)

    ax_prob.set_xlabel("Confidence (%)", fontsize=8.5); ax_prob.set_xlim(0, 112)
    ax_prob.set_title("(c) Class probability", fontsize=9.5, color=DARK, pad=6, loc="left")
    ax_prob.spines[["top", "right", "left"]].set_visible(False)
    ax_prob.tick_params(axis="y", left=False, labelsize=8.5)
    ax_prob.grid(color=GRAY_LT, lw=0.3, linestyle="--", axis="x", alpha=0.6, zorder=0)

    fig.text(0.5, 0.965,
             "(a) Grad-CAM spatial saliency maps on 4 representative frames",
             ha="center", fontsize=10.5, color=DARK)

    out_path = os.path.join(output_dir, "explain_{}.pdf".format(video_name))
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Figure saved → {}".format(out_path))
    return out_path


# ---------------------------------------------------------------------------
# CONFIG — edit here before running
# ---------------------------------------------------------------------------

# Directory containing tikharm_multimodal.py (None if same folder)
TIKHARM_DIR = None

# Video to analyze
VIDEO_PATH = "path/to/video.mp4"

# Saved model checkpoint
MODEL_PATH = "pe-video-wav2vec2-multimodal/best_model.pt"

# Fusion method used during training
FUSION_METHOD = "weighted"

# Class names in the exact order used during training
CLASSES = ["adult", "harmful", "safe", "suicide"]

# Must match max_frames used during training
NUM_FRAMES = 8

# Output directory for the PDF figure
OUTPUT_DIR = "explainability_output"

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if TIKHARM_DIR:
        sys.path.insert(0, TIKHARM_DIR)

    from tikharm_multimodal import Config, MultimodalClassifier, VideoLoader, AudioExtractor

    pred_label, probs = explain_video(
        video_path    = VIDEO_PATH,
        model_path    = MODEL_PATH,
        fusion_method = FUSION_METHOD,
        output_dir    = OUTPUT_DIR,
        num_frames    = NUM_FRAMES,
        classes       = CLASSES,
    )

    print("\n[RESULT] Prediction: {}".format(pred_label))
    for c, p in zip(CLASSES, probs):
        print("  {:20s}: {:.1f}%".format(c, p * 100))
