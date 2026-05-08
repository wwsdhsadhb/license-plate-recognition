"""
CRNN (Convolutional Recurrent Neural Network) for license plate recognition.
Architecture: CNN (VGG-like) + BiLSTM + CTC Loss
Reference: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (Shi et al., 2015)
"""

import torch
import torch.nn as nn


class VGGFeatureExtractor(nn.Module):
    """VGG-style CNN backbone for feature extraction."""

    def __init__(self, input_channel: int = 3, output_channel: int = 512):
        super().__init__()
        self.output_channel = [
            output_channel // 8,   # 64
            output_channel // 4,   # 128
            output_channel // 2,   # 256
            output_channel,        # 512
        ]

        self.ConvNet = nn.Sequential(
            # Stage 1
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Stage 2
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Stage 3
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Stage 4
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Stage 5
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for sequence modeling."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        return self.fc(recurrent)


class CRNN(nn.Module):
    """
    CRNN model for license plate character sequence recognition.

    Input:  (B, C, H, W) — grayscale or RGB image of a license plate
    Output: (T, B, num_classes) — log-softmax scores for CTC decoding
    """

    def __init__(
        self,
        input_channel: int = 3,
        img_height: int = 32,
        img_width: int = 100,
        num_classes: int = 68,
        hidden_size: int = 256,
    ):
        super().__init__()
        assert img_height % 16 == 0, "img_height must be divisible by 16"

        self.feature_extractor = VGGFeatureExtractor(input_channel, output_channel=512)

        # Compute the width of the feature map after CNN
        cnn_out_height = img_height // 16
        cnn_out_channels = 512

        self.rnn = nn.Sequential(
            BidirectionalLSTM(cnn_out_channels * cnn_out_height, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction: (B, C, H, W) -> (B, 512, H/16, W/4)
        features = self.feature_extractor(x)

        # Reshape for RNN: (B, C*H, W) -> (B, W, C*H)
        B, C, H, W = features.size()
        features = features.permute(0, 3, 1, 2)         # (B, W, C, H)
        features = features.reshape(B, W, C * H)        # (B, W, C*H)

        # RNN sequence modeling: (B, T, num_classes)
        output = self.rnn(features)

        # CTC expects (T, B, num_classes)
        output = output.permute(1, 0, 2)
        return self.log_softmax(output)


def build_crnn(num_classes: int = 68, pretrained_path: str = None) -> CRNN:
    """Build CRNN model, optionally loading pretrained weights."""
    model = CRNN(num_classes=num_classes)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[CRNN] Loaded weights from {pretrained_path}")
    return model
