import torch
import torch.nn as nn
import torch.nn.functional as F


class CCAM(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(CCAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1d = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm1d(channels // reduction_ratio),
            nn.ReLU(inplace=True)
        )

        self.expand_conv_h = nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1)
        self.expand_conv_w = nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros(1))


    def forward(self, x):
            b, c, h, w = x.size()

            F_prime = F.avg_pool2d(x, kernel_size=2, stride=2)

            F_flat = F_prime.view(b, c, -1)
            R, Z = F_flat, F_flat

            Sc = torch.bmm(Z, Z.transpose(1, 2))
            Sc = F.softmax(Sc, dim=-1)

            Mc = torch.bmm(Sc, R).view(b, c, h // 2, w // 2)

            Mc = F.interpolate(Mc, size=(h, w), mode='bilinear', align_corners=False)

            A_flat = x.view(b, c, -1)
            Mc_flat = Mc.view(b, c, -1)

            mean_A = torch.mean(A_flat, dim=-1, keepdim=True)
            mean_Mc = torch.mean(Mc_flat, dim=-1, keepdim=True)

            cov = torch.bmm((A_flat - mean_A), (Mc_flat - mean_Mc).transpose(1, 2)) / (h * w)

            Lc = Sc + cov

            Ec = torch.bmm(Lc, A_flat).view(b, c, h, w)

            attention_map = self.beta.to(x.device) * Ec + x
            out = x * attention_map

            return out, cov, Sc, Ec


