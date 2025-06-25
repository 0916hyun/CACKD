
import torch
import torch.nn.functional as F
from . import initialization as init
from .hub_mixin import SMPHubMixin
import torch.nn as nn

import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from custom_module.CCAM import *

class SegmentationModel(torch.nn.Module, SMPHubMixin):
    def initialize(self):
        # init.initialize_decoder_inp(self.decoder_inpainting)
        # init.initialize_decoder_seg(self.decoder_segmentation)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        # init.initialize_head_inp(self.segmentation_head_inpainting)
        # init.initialize_head_seg(self.segmentation_head_segmentation)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        """
        seg only 
        """
        skip_list = features
        deep_feature = features[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cam = CCAM(2048, 16).to(device)
        output = cam(deep_feature)
        features[-1] = output
        """
        seg only 
        """

        decoder_output = self.decoder(*features)

        final_result = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return final_result, labels

        return final_result, output, skip_list
        # return final_result, deep_feature
        # return final_result #CASE : res



    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x