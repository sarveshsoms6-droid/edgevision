"""Fine-tuning / transfer learning script for vehicle and pedestrian detection in EdgeVision.

Supports:
- Fine-tuning SSD or Faster R-CNN on traffic-specific custom dataset.
- Training detection heads while freezing backbone weights for efficient edge training.
- Extracting training frames from uploaded traffic videos.
- Saving PyTorch checkpoints for deployment and export.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)


def extract_frames_from_video(video_path: str, output_dir: str, max_frames: int = 100, stride: int = 5) -> List[str]:
    """Sample and save frames from an input video for dataset construction."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    saved = []
    f_idx = 0
    saved_count = 0
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % stride == 0:
            frame_filename = out_p / f"frame_{f_idx:05d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved.append(str(frame_filename))
            saved_count += 1
        f_idx += 1
    cap.release()
    print(f"Extracted {len(saved)} frames to {output_dir}")
    return saved


class TrafficDataset(Dataset):
    """Custom traffic dataset for fine-tuning detectors."""

    def __init__(self, image_paths: List[str], annotations: Optional[Dict] = None, img_size: int = 320):
        self.image_paths = image_paths
        self.annotations = annotations or {}
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_p = self.image_paths[idx]
        bgr = cv2.imread(img_p)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_p}")
        h_orig, w_orig = bgr.shape[:2]
        resized = cv2.resize(bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        # Retrieve annotations or dummy targets
        target: Dict[str, torch.Tensor] = {}
        if img_p in self.annotations:
            ann = self.annotations[img_p]
            boxes = np.array(ann.get("boxes", []), dtype=np.float32)
            # scale boxes to resized dimensions
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= (self.img_size / float(w_orig))
                boxes[:, [1, 3]] *= (self.img_size / float(h_orig))
                target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.tensor(ann.get("labels", []), dtype=torch.int64)
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)

        return tensor, target


def train_traffic_detector(
    model_name: str = "fasterrcnn_mobilenet_v3_large_320_fpn",
    dataset_dir: str = "dataset/images",
    epochs: int = 5,
    lr: float = 0.001,
    output_checkpoint: str = "models/fine_tuned_traffic.pth",
) -> None:
    """Train/fine-tune detection head on traffic images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting fine-tuning on device: {device} for {epochs} epochs")

    if "fasterrcnn" in model_name:
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    else:
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.to(device)
    model.train()

    # Collect images
    images = glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True) + \
             glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True)

    if not images:
        print(f"No training images found in {dataset_dir}. Please extract frames first.")
        return

    print(f"Found {len(images)} images in {dataset_dir} for training.")
    dataset = TrafficDataset(images[:100], img_size=320)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        epoch_loss = 0.0
        step_count = 0
        for batch_imgs, batch_targets in loader:
            batch_imgs = [img.to(device) for img in batch_imgs]
            batch_targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]

            # In training mode torchvision detectors return dict of loss components
            # If target has no boxes, skip step
            valid_targets = [t for t in batch_targets if len(t["boxes"]) > 0]
            if len(valid_targets) == 0:
                continue

            loss_dict = model(batch_imgs, batch_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            step_count += 1

        avg_loss = (epoch_loss / step_count) if step_count > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

    Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    print(f"Fine-tuned model checkpoint saved to: {output_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune EdgeVision traffic detector")
    parser.add_argument("--video", type=str, default=None, help="Extract training frames from video")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--model", type=str, default="fasterrcnn_mobilenet_v3_large_320_fpn")
    args = parser.parse_args()

    if args.video:
        extract_frames_from_video(args.video, "dataset/images", max_frames=50)
    train_traffic_detector(model_name=args.model, epochs=args.epochs)
