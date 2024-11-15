#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   StarNet.py
@Time      :   2024/06/12 19:43:33
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
import torch.nn as nn
from timm.layers import DropPath


# 定义 ConvBN 模块
class ConvBN(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        with_bn=True,
    ):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=not with_bn,
        )
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(out_planes)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        return x


# 定义 Block 模块
class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, n=1):
        super(StarBlock, self).__init__()
        self.n = n
        drop_path = 0.0
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, kernel_size=1)
        self.dwconv2 = ConvBN(
            dim, dim, kernel_size=7, padding=3, groups=dim, with_bn=False
        )
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        for _ in range(self.n):  # 使用循环来重复特定的操作n次
            x = self.dwconv(x)
            x1, x2 = self.f1(x), self.f2(x)
            x = self.act(x1) * x2
            x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)  # 确保残差连接仅应用一次，而不是n次
        return x


# 定义 StarNet 模块
class StarNet(nn.Module):
    def __init__(
        self,
        base_dim=32,
        depths=[3, 3, 12, 5],
        mlp_ratio=4,
        drop_path_rate=0.0,
        num_classes=1000,
    ):
        super(StarNet, self).__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6()
        )

        # 构建阶段
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # 第一阶段
        embed_dim = base_dim * 2**0
        down_sampler1 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks1 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[0])
        ]
        cur += depths[0]
        stage1 = nn.Sequential(down_sampler1, *blocks1)

        # 第二阶段
        embed_dim = base_dim * 2**1
        down_sampler2 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks2 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[1])
        ]
        cur += depths[1]
        stage2 = nn.Sequential(down_sampler2, *blocks2)

        # 第三阶段
        embed_dim = base_dim * 2**2
        down_sampler3 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks3 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[2])
        ]
        cur += depths[2]
        stage3 = nn.Sequential(down_sampler3, *blocks3)

        # 第四阶段
        embed_dim = base_dim * 2**3
        down_sampler4 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks4 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[3])
        ]
        cur += depths[3]
        stage4 = nn.Sequential(down_sampler4, *blocks4)

        self.stages = nn.ModuleList([stage1, stage2, stage3, stage4])
        # 分类头
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(self.norm(x))
        x = torch.flatten(x, 1)
        return self.head(x)


# 使用示例
if __name__ == "__main__":
    model = StarNet(
        base_dim=32,
        depths=[3, 3, 12, 5],
        mlp_ratio=4,
        drop_path_rate=0.1,
        num_classes=1000,
    )
    input_tensor = torch.randn(1, 3, 224, 224)  # 创建一个随机输入张量
    output = model(input_tensor)  # 前向传播
    print(output.shape)  # 输出分类结果
