import math
import torch
import torch.nn.functional as F
from layers import Conv2d, CategoricalConditionalBatchNorm2d 

def _upsample(x):
    upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
    return upsample(x)

def _downsample(x):
    return F.avg_pool2d(x, 2)

class ResBlock_G(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, k_size=3, pad=1,
                activation=F.relu, upsample=False, num_classes=0):
        super(ResBlock_G, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        if hidden_channels is None:
            hidden_channels = out_channels
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(in_channels, hidden_channels, k_size, 1, pad)
        self.conv2 = torch.nn.Conv2d(hidden_channels, out_channels, k_size, 1, pad)
        if self.num_classes > 0:
            self.bn1 = CategoricalConditionalBatchNorm2d(self.num_classes, in_channels)
            self.bn2 = CategoricalConditionalBatchNorm2d(self.num_classes, hidden_channels)
        else:
            self.bn1 = torch.nn.BatchNorm2d(in_channels)
            self.bn2 = torch.nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.short_cut = torch.nn.Conv2d(in_channels, out_channels, 1)
        self._initialize()

    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=math.sqrt(2))
        if self.learnable_sc:
            torch.nn.init.xavier_uniform_(self.short_cut.weight, gain=1)

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = _upsample(x)
            x = self.short_cut(x)
            return x
        else:
            return x

    def residual(self, x, y=None):
        if y is not None:
            x = self.bn1(x, y)
        else:
            x = self.bn1(x)
        x = self.activation(x)
        if self.upsample:
            x = _upsample(x)
        x = self.conv1(x)

        if y is not None:
            x = self.bn2(x, y)
        else:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x, y=None):
        return self.shortcut(x) + self.residual(x, y)

class ResBlock_D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, k_size=3, pad=1,
                activation=F.relu, downsample=False):
        super(ResBlock_D, self).__init__()

        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        if hidden_channels is None:
            hidden_channels = in_channels
        self.conv1 = Conv2d(in_channels, hidden_channels, k_size, 1, pad)
        self.conv2 = Conv2d(hidden_channels, out_channels, k_size, 1, pad)
        if self.learnable_sc:
            self.short_cut = Conv2d(in_channels, out_channels, 1, 1, 0)
        self._initialize()

    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=math.sqrt(2))
        if self.learnable_sc:
            torch.nn.init.xavier_uniform_(self.short_cut.weight, gain=1)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.short_cut(x)
        if self.downsample:
            x = _downsample(x)
        return x

    def residual(self, x):
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = _downsample(x)
        return x

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

class ResBlock_D_opt(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, pad=1, activation=F.relu):
        super(ResBlock_D_opt, self).__init__()

        self.activation = activation
        self.conv1 = Conv2d(in_channels, out_channels, k_size, 1, pad)
        self.conv2 = Conv2d(out_channels, out_channels, k_size, 1, pad)
        self.short_cut = Conv2d(in_channels, out_channels, 1, 1, 0)
        self._initialize()

    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.short_cut.weight, gain=1)

    def shortcut(self, x):
        return self.short_cut(_downsample(x))

    def residual(self, x):
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = _downsample(x)
        return x

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)



