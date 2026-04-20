from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Собственная сверточная сеть для классификации дорожных знаков GTSRB.

    Архитектура: 4 сверточных блока вида (Conv-BN-ReLU) x2 + MaxPool,
    затем adaptive average pooling и классификатор из двух Linear слоёв.

    Рассчитана на входное разрешение 48-64 пикселя (знаки маленькие,
    больше не нужно) и обучение с нуля.
    """

    def __init__(self, num_classes: int = 43, dropout: float = 0.4):
        super().__init__()

        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2.0),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
