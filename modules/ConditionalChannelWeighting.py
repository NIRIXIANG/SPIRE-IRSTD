import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

def optimized_channel_shuffle(x, groups):
    """高效通道洗牌，减少 `contiguous()` 操作"""
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    return x.reshape(batchsize, groups, channels_per_group, height, width) \
            .permute(0, 2, 1, 3, 4) \
            .reshape(batchsize, num_channels, height, width)

class OptimizedSpatialWeighting(nn.Module):
    """优化后的空间注意力模块"""
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(1, channels // ratio)  # 避免 `0` 通道
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.global_avgpool(x))

class OptimizedConditionalChannelWeighting(nn.Module):
    def __init__(self, in_channels, stride, reduce_ratio, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        if isinstance(in_channels, int):
            in_channels = [in_channels]

        branch_channels = [max(1, channel // 2) for channel in in_channels]

        self.depthwise_convs = nn.Conv2d(
            in_channels=sum(branch_channels),
            out_channels=sum(branch_channels),
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=sum(branch_channels),
            bias=False
        )
        self.bn = nn.BatchNorm2d(sum(branch_channels))

        self.spatial_weighting = nn.Sequential(*[
            OptimizedSpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):
        def _inner_forward(x):
            # **直接在 `Tensor` 上 chunk，不再用 `list`**
            x1, x2 = x.chunk(2, dim=1)

            x2 = self.depthwise_convs(x2)
            x2 = self.bn(x2)
            x2 = self.spatial_weighting(x2)

            # **拼接回原通道**
            out = torch.cat([x1, x2], dim=1)

            # **通道洗牌**
            out = optimized_channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)

