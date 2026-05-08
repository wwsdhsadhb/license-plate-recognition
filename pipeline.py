"""
License plate detector based on a lightweight YOLOv5-inspired architecture.
For full YOLOv5 usage, install: pip install ultralytics
This module provides a wrapper that can use either:
  1. ultralytics YOLOv5 (if installed)
  2. A lightweight custom detector as fallback
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Lightweight custom detector (fallback)
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightDetector(nn.Module):
    """
    Lightweight single-stage detector for license plate localization.
    Input:  (B, 3, 320, 320)
    Output: List of (x1, y1, x2, y2, confidence) per image
    """

    def __init__(self, num_anchors: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNReLU(3,   32,  3, 2, 1),   # 160
            ConvBNReLU(32,  64,  3, 2, 1),   # 80
            ConvBNReLU(64,  128, 3, 2, 1),   # 40
            ConvBNReLU(128, 256, 3, 2, 1),   # 20
            ConvBNReLU(256, 512, 3, 2, 1),   # 10
        )
        # Detection head: predict (cx, cy, w, h, conf) per anchor
        self.head = nn.Conv2d(512, num_anchors * 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class LicensePlateDetector:
    """
    High-level wrapper for license plate detection.
    Supports YOLOv5 (ultralytics) or the built-in LightDetector.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_ultralytics: bool = False,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device(device)
        self.use_ultralytics = use_ultralytics

        if use_ultralytics:
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights_path or "yolov5s.pt")
                print("[Detector] Using ultralytics YOLOv5")
            except ImportError:
                print("[Detector] ultralytics not found, falling back to LightDetector")
                self.use_ultralytics = False

        if not use_ultralytics:
            self.model = LightDetector().to(self.device)
            if weights_path:
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )
                print(f"[Detector] Loaded weights from {weights_path}")
            self.model.eval()

    def preprocess(self, image: np.ndarray, size: int = 320) -> Tuple[torch.Tensor, float, float]:
        """Resize image and convert to tensor. Returns tensor + scale factors."""
        h, w = image.shape[:2]
        resized = cv2.resize(image, (size, size))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device), w / size, h / size

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run detection on a BGR image (OpenCV format).
        Returns list of (x1, y1, x2, y2, confidence) in original image coordinates.
        """
        if self.use_ultralytics:
            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    boxes.append((x1, y1, x2, y2, conf))
            return boxes

        # LightDetector inference (demo — returns entire image as single detection)
        tensor, sx, sy = self.preprocess(image)
        with torch.no_grad():
            _ = self.model(tensor)
        h, w = image.shape[:2]
        # Placeholder: return full image crop (replace with real NMS post-processing)
        return [(0, 0, w, h, 1.0)]

    def crop_plates(
        self, image: np.ndarray, padding: float = 0.05
    ) -> List[np.ndarray]:
        """Detect and crop all license plate regions from an image."""
        detections = self.detect(image)
        crops = []
        h, w = image.shape[:2]
        for x1, y1, x2, y2, conf in detections:
            if conf < self.conf_threshold:
                continue
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crops.append(image[y1:y2, x1:x2])
        return crops
