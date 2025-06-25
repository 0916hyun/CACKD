import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md


class SegmentationDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,  # 인페인팅 결과와 skip 연결 채널 합산
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        # Upsample to match the size of the skip connection
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # If skip connection exists, concatenate it along the channel dimension
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        # Apply two convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class SegmentationUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        # Ensure that the number of decoder channels matches the number of blocks
        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provided `decoder_channels` for {len(decoder_channels)} blocks."
            )

        # Reverse the encoder channels to start from the deepest layers
        encoder_channels = encoder_channels[::-1]

        # Define input channels for each decoder block
        in_channels = [3] + list(decoder_channels[:-1])  # 첫 번째 입력은 인페인팅 결과로 고정
        skip_channels = tuple(encoder_channels[1:]) + (0,)  # 스킵 연결을 받아들이는 채널 수
        out_channels = decoder_channels

        # Create decoder blocks
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.blocks = nn.ModuleList([
            SegmentationDecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])

    def forward(self, *features):
        features = features[1:]  # Remove the first feature (same resolution)
        features = features[::-1]  # Reverse channels to start from deepest encoder feature

        x = features[0]  # 인페인팅 결과

        skips = features[1:]  # Skip connections

        # Pass through each decoder block
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

