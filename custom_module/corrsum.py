import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sc(x):
    B, C, H, W = x.shape
    x_flat = F.normalize(x.view(B, C, -1), p=2, dim=-1)
    sc_matrix = torch.bmm(x_flat, x_flat.transpose(1, 2))


    # min_val = sc_matrix.min(dim=-1, keepdim=True)[0]
    # max_val = sc_matrix.max(dim=-1, keepdim=True)[0]
    # sc_matrix_norm = (sc_matrix - min_val) / (max_val - min_val + 1e-8)
    #
    # return sc_matrix_norm
    return sc_matrix


class SkipScAggregator(nn.Module):


    def __init__(self, skip_in_channels, base_channel=256):

        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(in_c, base_channel, kernel_size=1, bias=False)
            for in_c in skip_in_channels
        ])

        self.alphas = nn.Parameter(torch.ones(len(skip_in_channels)))

    def forward(self, skip_features):

        sum_sc = None
        num_stages = len(skip_features)

        eps = 1e-8

        for i, feat in enumerate(skip_features):

            unified_feat = self.convs[i](feat)

            sc_matrix = compute_sc(unified_feat)

            if sum_sc is None:
                sum_sc = self.alphas[i] * sc_matrix
            else:
                sum_sc = sum_sc + self.alphas[i] * sc_matrix

        alpha_sum = torch.sum(self.alphas) + eps
        final_sc = sum_sc / alpha_sum

        return final_sc
