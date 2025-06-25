# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from segmentation_models_pytorch.base import modules as md
#
#
# class DecoderBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         skip_channels,
#         out_channels,
#         use_batchnorm=True,
#         attention_type=None,
#     ):
#         super().__init__()
#         self.conv1 = md.Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention1 = md.Attention(
#             attention_type, in_channels=in_channels + skip_channels
#         )
#         self.conv2 = md.Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention2 = md.Attention(attention_type, in_channels=out_channels)
#
#     def forward(self, x, skip=None):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#             x = self.attention1(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.attention2(x)
#         return x
#
#
# class CenterBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels, use_batchnorm=True):
#         conv1 = md.Conv2dReLU(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         conv2 = md.Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         super().__init__(conv1, conv2)
#
#
# # class UnetPlusPlusDecoder(nn.Module):
# #     def __init__(
# #         self,
# #         encoder_channels,
# #         decoder_channels,
# #         n_blocks=5,
# #         use_batchnorm=True,
# #         attention_type=None,
# #         center=False,
# #     ):
# #         super().__init__()
# #
# #         if n_blocks != len(decoder_channels):
# #             raise ValueError(
# #                 "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
# #                     n_blocks, len(decoder_channels)
# #                 )
# #             )
# #
# #         # remove first skip with same spatial resolution
# #         encoder_channels = encoder_channels[1:]
# #         # reverse channels to start from head of encoder
# #         encoder_channels = encoder_channels[::-1]
# #
# #         # computing blocks input and output channels
# #         head_channels = encoder_channels[0]
# #         self.in_channels = [head_channels] + list(decoder_channels[:-1])
# #         self.skip_channels = list(encoder_channels[1:]) + [0]
# #         self.out_channels = decoder_channels
# #         if center:
# #             self.center = CenterBlock(
# #                 head_channels, head_channels, use_batchnorm=use_batchnorm
# #             )
# #         else:
# #             self.center = nn.Identity()
# #
# #         # combine decoder keyword arguments
# #         kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
# #
# #         blocks = {}
# #         for layer_idx in range(len(self.in_channels) - 1):
# #             for depth_idx in range(layer_idx + 1):
# #                 if depth_idx == 0:
# #                     in_ch = self.in_channels[layer_idx]
# #                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
# #                     out_ch = self.out_channels[layer_idx]
# #                 else:
# #                     out_ch = self.skip_channels[layer_idx]
# #                     skip_ch = self.skip_channels[layer_idx] * (
# #                         layer_idx + 1 - depth_idx
# #                     )
# #                     in_ch = self.skip_channels[layer_idx - 1]
# #                 blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
# #                     in_ch, skip_ch, out_ch, **kwargs
# #                 )
# #         blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
# #             self.in_channels[-1], 0, self.out_channels[-1], **kwargs
# #         )
# #         self.blocks = nn.ModuleDict(blocks)
# #         self.depth = len(self.in_channels) - 1
# #
# #     def forward(self, *features):
# #         features = features[1:]  # remove first skip with same spatial resolution
# #         features = features[::-1]  # reverse channels to start from head of encoder
# #         # start building dense connections
# #         dense_x = {}
# #         saved_features = {} #중간 피쳐 저장용 추가본이에요
# #         for layer_idx in range(len(self.in_channels) - 1):
# #             for depth_idx in range(self.depth - layer_idx):
# #                 if layer_idx == 0:
# #                     output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
# #                         features[depth_idx], features[depth_idx + 1]
# #                     )
# #                     dense_x[f"x_{depth_idx}_{depth_idx}"] = output
# #                 else:
# #                     dense_l_i = depth_idx + layer_idx
# #                     cat_features = [
# #                         dense_x[f"x_{idx}_{dense_l_i}"]
# #                         for idx in range(depth_idx + 1, dense_l_i + 1)
# #                     ]
# #                     cat_features = torch.cat(
# #                         cat_features + [features[dense_l_i + 1]], dim=1
# #                     )
# #                     dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
# #                         f"x_{depth_idx}_{dense_l_i}"
# #                     ](dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features)
# #
# #         saved_features["x_4_0"] = dense_x.get("x_4_0") #중간 피쳐 저장용 추가본이에요
# #         saved_features["x_3_1"] = dense_x.get("x_3_1") #중간 피쳐 저장용 추가본이에요
# #         saved_features["x_2_2"] = dense_x.get("x_2_2") #중간 피쳐 저장용 추가본이에요
# #         saved_features["x_1_3"] = dense_x.get("x_1_3") #중간 피쳐 저장용 추가본이에요
# #         saved_features["x_0_4"] = dense_x.get("x_0_4") #중간 피쳐 저장용 추가본이에요
# #
# #         dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
# #             dense_x[f"x_{0}_{self.depth-1}"]
# #         )
# #         saved_features["final_output"] = dense_x[f"x_{0}_{self.depth}"] #중간 피쳐 저장용 추가본이에요
# #
# #         return dense_x[f"x_{0}_{self.depth}"], saved_features #중간 피쳐 저장용 추가본이에요
#
#
#
# class UnetPlusPlusDecoder(nn.Module):
#     def __init__(
#         self,
#         encoder_channels,
#         decoder_channels,
#         n_blocks=5,
#         use_batchnorm=True,
#         attention_type=None,
#         center=False,
#     ):
#         super().__init__()
#
#         if n_blocks != len(decoder_channels):
#             raise ValueError(
#                 "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
#                     n_blocks, len(decoder_channels)
#                 )
#             )
#
#         # remove first skip with same spatial resolution
#         encoder_channels = encoder_channels[1:]
#         # reverse channels to start from head of encoder
#         encoder_channels = encoder_channels[::-1]
#
#         # computing blocks input and output channels
#         head_channels = encoder_channels[0]
#         self.in_channels = [head_channels] + list(decoder_channels[:-1])
#         self.skip_channels = list(encoder_channels[1:]) + [0]
#         self.out_channels = decoder_channels
#         if center:
#             self.center = CenterBlock(
#                 head_channels, head_channels, use_batchnorm=use_batchnorm
#             )
#         else:
#             self.center = nn.Identity()
#
#         # combine decoder keyword arguments
#         kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
#
#         blocks = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(layer_idx + 1):
#                 if depth_idx == 0:
#                     in_ch = self.in_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
#                     out_ch = self.out_channels[layer_idx]
#                 else:
#                     out_ch = self.skip_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (
#                         layer_idx + 1 - depth_idx
#                     )
#                     in_ch = self.skip_channels[layer_idx - 1]
#                 blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
#                     in_ch, skip_ch, out_ch, **kwargs
#                 )
#         blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
#             self.in_channels[-1], 0, self.out_channels[-1], **kwargs
#         )
#         self.blocks = nn.ModuleDict(blocks)
#         self.depth = len(self.in_channels) - 1
#
#     def forward(self, *features):
#         features = features[1:]  # remove first skip with same spatial resolution
#         features = features[::-1]  # reverse channels to start from head of encoder
#         # start building dense connections
#         dense_x = {}
#         saved_features = {}  # 중간 피처 저장용
#
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(self.depth - layer_idx):
#                 if layer_idx == 0:
#                     output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
#                         features[depth_idx], features[depth_idx + 1]
#                     )
#                     dense_x[f"x_{depth_idx}_{depth_idx}"] = output
#                 else:
#                     dense_l_i = depth_idx + layer_idx
#                     cat_features = [
#                         dense_x[f"x_{idx}_{dense_l_i}"]
#                         for idx in range(depth_idx + 1, dense_l_i + 1)
#                     ]
#                     cat_features = torch.cat(
#                         cat_features + [features[dense_l_i + 1]], dim=1
#                     )
#                     dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
#                         f"x_{depth_idx}_{dense_l_i}"
#                     ](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
#
#         # 필요한 중간 피처를 저장
#         saved_features["x_4_0"] = dense_x.get("x_4_0")
#         saved_features["x_3_1"] = dense_x.get("x_3_1")
#         saved_features["x_2_2"] = dense_x.get("x_2_2")
#         saved_features["x_1_3"] = dense_x.get("x_1_3")
#         saved_features["x_0_4"] = dense_x.get("x_0_4")
#
#         # 마지막 디코더 레이어에서 최종 출력을 생성
#         final_output = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth - 1}"])
#
#         # dense_x의 키와 해당 값의 shape을 출력
#         # for key in dense_x:
#         #     print(f"Key: {key}, Shape: {dense_x[key].shape}")
#
#         # 최종 출력과 동일하게 'final_output'을 저장
#         saved_features["final_output"] = final_output
#         features_return = saved_features["final_output"]
#         # 리턴 값으로 최종 출력과 저장된 중간 피처들을 함께 반환
#         return final_output, features_return
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
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

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, skip_ch, out_ch, **kwargs
                )
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth-1}"]
        )
        return dense_x[f"x_{0}_{self.depth}"]
