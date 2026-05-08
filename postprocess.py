"""
Visualization utilities: draw bounding boxes and recognition results on images.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional


def draw_results(
    image: np.ndarray,
    results: List[Dict[str, Any]],
    font_scale: float = 0.7,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes and recognition text on an image.

    Args:
        image:   BGR numpy array
        results: list of dicts from LicensePlateRecognizer.run()

    Returns:
        Annotated BGR image
    """
    vis = image.copy()

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        plate = r["plate"]
        conf  = r["confidence"]
        valid = r["valid"]

        color = (0, 200, 0) if valid else (0, 80, 255)

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label = f"{plate}  {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        bg_y1 = max(0, y1 - lh - 8)
        cv2.rectangle(vis, (x1, bg_y1), (x1 + lw + 4, y1), color, -1)

        cv2.putText(
            vis, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return vis


def save_visualization(
    image: np.ndarray,
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    vis = draw_results(image, results)
    cv2.imwrite(output_path, vis)
    print(f"[Visualize] Saved to {output_path}")


def show_image(image: np.ndarray, title: str = "License Plate Recognition") -> None:
    """Display image in a window (requires GUI environment)."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
