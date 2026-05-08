"""
Single-image inference entry point.

Usage:
    python inference.py --image path/to/car.jpg
    python inference.py --image path/to/car.jpg --show
    python inference.py --image path/to/car.jpg --weights weights/crnn_best.pth
"""

import argparse
import os
import cv2

from models.pipeline import LicensePlateRecognizer
from utils.visualize import draw_results, show_image, save_visualization
from utils.postprocess import format_result


def parse_args():
    p = argparse.ArgumentParser(description="License Plate Recognition Inference")
    p.add_argument("--image",    type=str, required=True, help="Path to input image")
    p.add_argument("--weights",  type=str, default=None,  help="Path to CRNN weights (.pth)")
    p.add_argument("--device",   type=str, default="cpu")
    p.add_argument("--show",     action="store_true",     help="Display result window")
    p.add_argument("--save",     type=str, default=None,  help="Save output image to path")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[Error] Image not found: {args.image}")
        return

    # Build pipeline
    recognizer = LicensePlateRecognizer(
        recognizer_weights=args.weights,
        device=args.device,
    )

    # Run
    image = cv2.imread(args.image)
    results = recognizer.run(image)

    # Print results
    print(f"\n{'─'*40}")
    print(f"  Image : {args.image}")
    if results:
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {format_result(r['plate'], r['confidence'], r['valid'])}")
            print(f"      Latency: {r['latency_ms']} ms  |  BBox: {r['bbox']}")
    else:
        print("  No license plate detected.")
    print(f"{'─'*40}\n")

    # Visualize
    vis = draw_results(image, results)
    if args.save:
        save_visualization(image, results, args.save)
    if args.show:
        show_image(vis)


if __name__ == "__main__":
    main()
