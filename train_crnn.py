"""
Training script for the CRNN license plate recognizer.

Usage:
    python scripts/train_crnn.py --data_dir ./data --epochs 50 --batch_size 64
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.crnn import build_crnn
from utils.dataset import build_dataloader
from utils.ctc_decoder import CTCDecoder, BLANK_IDX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir",    type=str, default="./weights")
    p.add_argument("--resume",      type=str, default=None)
    p.add_argument("--device",      type=str, default="cpu")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels, label_lens in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        # Forward
        log_probs = model(images)              # (T, B, C)
        T, B, C = log_probs.shape
        input_lens = torch.full((B,), T, dtype=torch.long, device=device)

        # CTC loss requires flat labels
        flat_labels = labels.flatten()
        flat_labels = flat_labels[flat_labels != 0]   # remove padding

        loss = criterion(log_probs, flat_labels, input_lens, label_lens)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, decoder, device):
    model.eval()
    correct = total = 0

    for images, labels, label_lens in tqdm(loader, desc="Eval ", leave=False):
        images = images.to(device)
        log_probs = model(images)

        preds = decoder.decode_batch(log_probs)
        B = images.size(0)
        for i in range(B):
            # Decode ground truth
            gt_len = label_lens[i].item()
            gt_indices = labels[i, :gt_len].tolist()
            from utils.ctc_decoder import IDX2CHAR
            gt_text = "".join(IDX2CHAR.get(idx, "?") for idx in gt_indices)
            pred_text = preds[i][0]
            if pred_text == gt_text:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    train_loader = build_dataloader(
        os.path.join(args.data_dir, "train"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )
    val_loader = build_dataloader(
        os.path.join(args.data_dir, "val"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        shuffle=False,
    )

    model = build_crnn(num_classes=68, pretrained_path=args.resume).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    decoder = CTCDecoder()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc    = evaluate(model, val_loader, decoder, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, "crnn_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (acc={best_acc:.4f}) to {save_path}")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
