"""SPIRE-Net: IR small-target heatmap head (checkpoint-compatible with P2P_Lite_v7_6)."""
import torch
import torch.nn as nn

from modules.ConditionalChannelWeighting import OptimizedConditionalChannelWeighting
from modules.LA_OCCW import LA_OptimizedConditionalChannelWeighting

BN_MOMENTUM = 0.1


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, -1, h, w)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        assert stride in (1, 2)
        branch_channels = out_channels // 2

        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        in_ch = branch_channels if stride == 1 else in_channels
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, stride=stride, padding=1,
                      groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels, momentum=BN_MOMENTUM),
            nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        return channel_shuffle(out, 2)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class OCCW_StageModule(nn.Module):
    """Multi-scale fusion stage (OCCW + shuffle)."""

    def __init__(self, input_branches, output_branches, c):
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches
        self.branches = nn.ModuleList()

        for i in range(self.input_branches):
            w = c * (2 ** i)
            self.branches.append(nn.Sequential(
                OptimizedConditionalChannelWeighting(w, 1, 4, with_cp=False),
                LA_OptimizedConditionalChannelWeighting(w, 1, 4, with_cp=False),
                ShuffleUnit(in_channels=w, out_channels=w, stride=1),
            ))

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            fuse_row = nn.ModuleList()
            for j in range(self.input_branches):
                if i == j:
                    fuse_row.append(nn.Identity())
                elif i < j:
                    fuse_row.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), 1, bias=False),
                        nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest'),
                    ))
                else:
                    ops = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            ops.append(nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** i), 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            ))
                        else:
                            ops.append(nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_row.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_row)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fused = []
        for i, fuse_row in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_row[0](x[0])
            for j in range(self.input_branches):
                y = y + (x[j] if i == j else fuse_row[j](x[j]))
            x_fused.append(self.relu(y))
        return x_fused


class SPIRENet(nn.Module):
    """
    Heatmap network for IR point targets.
    State dict keys match former ``P2P_Lite_v7_6`` (load with ``strict=True``).
    """

    def __init__(self, base_channel: int = 32, num_joints: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            nn.Conv2d(64, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            ShuffleUnit(256, 256, stride=1),
            ShuffleUnit(256, 256, stride=1),
            ShuffleUnit(256, 256, stride=1),
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ),
        ])

        self.stage2 = OCCW_StageModule(input_branches=1, output_branches=1, c=base_channel)
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential(
            OCCW_StageModule(input_branches=1, output_branches=1, c=base_channel),
        )
        self.final_layer = nn.Conv2d(base_channel, num_joints, 1, stride=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.final_layer(x[0])
