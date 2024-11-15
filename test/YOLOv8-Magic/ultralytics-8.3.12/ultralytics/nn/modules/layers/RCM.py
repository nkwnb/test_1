#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   RCM.py
@Time      :   2024/09/12 20:58:03
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple


class ConvMlp(nn.Module):
    """使用 1x1 卷积保持空间维度的 MLP"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# rectangular self-calibration attention (RCA)
class RCA(nn.Module):
    def __init__(
        self,
        inp,
        kernel_size=1,
        ratio=1,
        band_kernel_size=11,
        dw_size=(1, 1),
        padding=(0, 0),
        stride=1,
        square_kernel_size=2,
        relu=True,
    ):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(
            inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(
                inp,
                gc,
                kernel_size=(1, band_kernel_size),
                padding=(0, band_kernel_size // 2),
                groups=gc,
            ),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                gc,
                inp,
                kernel_size=(band_kernel_size, 1),
                padding=(band_kernel_size // 2, 0),
                groups=gc,
            ),
            nn.Sigmoid(),
        )

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather)  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc

        return out


# Rectangular Self-Calibration Module (RCM)
class RCM(nn.Module):
    """MetaNeXtBlock 块
    参数:
        dim (int): 输入通道数.
        drop_path (float): 随机深度率。默认: 0.0
        ls_init_value (float): 层级比例初始化值。默认: 1e-6.
    """

    def __init__(
        self,
        dim,
        token_mixer=RCA,
        norm_layer=nn.BatchNorm2d,
        mlp_layer=ConvMlp,
        mlp_ratio=2,
        act_layer=nn.GELU,
        ls_init_value=1e-6,
        drop_path=0.0,
        dw_size=11,
        square_kernel_size=3,
        ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(
            dim,
            band_kernel_size=dw_size,
            square_kernel_size=square_kernel_size,
            ratio=ratio,
        )
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


if __name__ == "__main__":

    input = torch.randn(1, 64, 32, 32)
    block = RCM(dim=64)
    output = block(input)
    print(input.size())
    print(output.size())