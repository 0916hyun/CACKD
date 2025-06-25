from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
import torch
import torch.nn as nn

from .decoder import UnetDecoder
from .seg_decoder import SegmentationUnetDecoder
#
# class CALayer(nn.Module):
#     """Channel Attention Layer (CALayer)"""
#
#     def __init__(self, channel):
#         super(CALayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
#             nn.GELU(),
#             nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.ca(y)
#         return x * y
#
#
# class ADCARB(nn.Module):
#     """Attention-based Deformable Convolutional and Recurrent Block (ADCARB)"""
#
#     def __init__(self, dim):
#         super(ADCARB, self).__init__()
#
#         # Dilated convolutions with different dilation rates
#         self.conv_dilated_1 = nn.Conv2d(dim, dim // 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
#                                         padding_mode='reflect')
#         self.conv_dilated_4 = nn.Conv2d(dim, dim // 16, kernel_size=3, stride=1, padding=4, dilation=4, bias=True,
#                                         padding_mode='reflect')
#         self.conv_dilated_16 = nn.Conv2d(dim, dim // 16, kernel_size=3, stride=1, padding=16, dilation=16, bias=True,
#                                          padding_mode='reflect')
#
#         # 3x3 convolutions for fusion and final output
#         self.conv_fuse = nn.Conv2d(3 * (dim // 16), dim, kernel_size=3, stride=1, padding=1, bias=True,
#                                    padding_mode='reflect')
#         self.conv_dest = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
#         self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
#
#         # Activation, attention, and normalization
#         self.activation = nn.GELU()
#         self.calayer = CALayer(dim)
#         self.InstanceNorm = nn.InstanceNorm2d(dim, affine=False)
#
#         self._init_weight()
#
#     def forward(self, x):
#         # Dilated convolutions with different dilation rates
#         x_di1 = self.activation(self.conv_dilated_1(x))
#         x_di4 = self.activation(self.conv_dilated_4(x))
#         x_di16 = self.activation(self.conv_dilated_16(x))
#
#         # Apply attention mask
#         mask = self.calayer(self.conv_dest(x))
#         mask = torch.sigmoid(mask)
#
#         # Fuse the dilated convolution outputs
#         x_c = self.conv_fuse(torch.cat((x_di1, x_di4, x_di16), dim=1))
#         x_c = self.activation(self.InstanceNorm(x_c))
#         x_c = self.InstanceNorm(self.conv3x3(x_c))
#
#         # Apply the attention mask
#         x_c = x * (1 - mask) + x_c * mask
#
#         # Final attention layer
#         x_c = self.calayer(x_c)
#         x_c += x
#
#         return x_c
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#                 nn.init.constant_(m.bias, 0)

class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes_inpainting: int = 3,  # Output channels for inpainting (RGB)
        classes_segmentation: int = 12,  # Output channels for segmentation (binary/multi-class)
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        # self.adcarb1 = ADCARB(self.encoder.out_channels[-1])
        # self.adcarb2 = ADCARB(self.encoder.out_channels[-1])
        # self.adcarb3 = ADCARB(self.encoder.out_channels[-1])

        self.decoder_inpainting = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        # Second decoder for segmentation task
        self.decoder_segmentation = SegmentationUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        # Inpainting head
        self.segmentation_head_inpainting = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_inpainting,  # RGB for inpainting
            activation=activation,
            kernel_size=3,
        )

        # Segmentation head
        self.segmentation_head_segmentation = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_segmentation,  # Segmentation output
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

