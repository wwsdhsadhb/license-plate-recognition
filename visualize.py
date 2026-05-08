"""
CTC (Connectionist Temporal Classification) greedy decoder.
Converts log-probability sequences from CRNN output into character strings.
"""

import torch
import numpy as np
from typing import Tuple, List

# -----------------------------------------------------------------------
# Character set for Chinese license plates
# Format: [blank] + 省份简称 + 字母 + 数字
# Total: 1 + 31 + 24 + 10 = 66, padded to 68 with special tokens
# -----------------------------------------------------------------------

PROVINCES = [
    "京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘",
    "皖", "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋",
    "蒙", "陕", "吉", "闽", "贵", "粤", "川", "青", "琼", "宁",
    "藏",
]

LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")   # no I, O (24 chars)
DIGITS  = list("0123456789")

CHARS = ["<blank>"] + PROVINCES + LETTERS + DIGITS + ["<unk>"]
# Pad to 68 if needed
while len(CHARS) < 68:
    CHARS.append(f"<pad{len(CHARS)}>")

CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

BLANK_IDX = 0


class CTCDecoder:
    """Greedy CTC decoder (best-path decoding)."""

    def __init__(self, blank_idx: int = BLANK_IDX):
        self.blank_idx = blank_idx

    def decode(
        self, log_probs: torch.Tensor
    ) -> Tuple[str, float]:
        """
        Decode a single sequence.

        Args:
            log_probs: (T, 1, C) log-softmax tensor from CRNN

        Returns:
            (decoded_string, mean_confidence)
        """
        # Greedy: argmax at each time step
        probs = torch.exp(log_probs[:, 0, :])   # (T, C)
        indices = torch.argmax(probs, dim=1).cpu().numpy()   # (T,)
        confidences = probs.max(dim=1).values.cpu().detach().numpy()

        # Collapse repeated chars and remove blanks
        chars = []
        conf_list = []
        prev = None
        for idx, conf in zip(indices, confidences):
            if idx != prev:
                if idx != self.blank_idx:
                    chars.append(IDX2CHAR.get(idx, "?"))
                    conf_list.append(float(conf))
                prev = idx

        text = "".join(chars)
        mean_conf = float(np.mean(conf_list)) if conf_list else 0.0
        return text, mean_conf

    def decode_batch(
        self, log_probs: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Decode a batch.

        Args:
            log_probs: (T, B, C)

        Returns:
            List of (decoded_string, confidence) for each item in batch
        """
        B = log_probs.size(1)
        return [
            self.decode(log_probs[:, i:i+1, :]) for i in range(B)
        ]
