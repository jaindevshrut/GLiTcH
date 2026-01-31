import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: tuple = (3, 3), s: tuple = (1, 1), p: tuple = (1, 1), se: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-3, momentum=0.01)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.se = SqueezeExcite(out_ch) if se else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        x = self.se(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple = (1, 1), expand: int = 2, se: bool = True, t_kernel: int = 3, f_kernel: int = 3):
        super().__init__()
        mid = in_ch * expand
        self.use_residual = stride == (1, 1) and in_ch == out_ch
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.dw_t = nn.Conv2d(mid, mid, kernel_size=(1, t_kernel), stride=stride, padding=(0, t_kernel // 2), groups=mid, bias=False)
        self.dw_f = nn.Conv2d(mid, mid, kernel_size=(f_kernel, 1), stride=(1, 1), padding=(f_kernel // 2, 0), groups=mid, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid, eps=1e-3, momentum=0.01)
        self.se = SqueezeExcite(mid) if se else nn.Identity()
        self.pw = nn.Conv2d(mid, out_ch, 1, 1, 0, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)

    def forward(self, x):
        y = self.expand(x)
        y = self.dw_t(y)
        y = self.dw_f(y)
        y = F.relu(self.bn_dw(y), inplace=True)
        y = self.se(y)
        y = self.bn_pw(self.pw(y))
        if self.use_residual:
            y = y + x
        return F.relu(y, inplace=True)


class StatPool(nn.Module):
    def __init__(self, dims: tuple = (2, 3)):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        m = x.mean(dim=self.dims)
        s = x.std(dim=self.dims)
        return torch.cat([m, s], dim=1)


class LightweightDeepfakeDetector(nn.Module):
    def __init__(self, n_mels: int = 64, base_ch: int = 16, num_classes: int = 2):
        super().__init__()
        self.n_mels = n_mels
        c1 = base_ch
        c2 = int(base_ch * 1.5)
        c3 = base_ch * 2
        c4 = int(base_ch * 3)
        c5 = base_ch * 4
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(c1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(c1, c1, k=(3, 3), s=(1, 1), p=(1, 1), se=True),
            InvertedResidual(c1, c2, stride=(1, 2), expand=2, se=True),
        )
        self.block2 = nn.Sequential(
            InvertedResidual(c2, c2, stride=(1, 1), expand=2, se=True),
            InvertedResidual(c2, c3, stride=(1, 2), expand=2, se=True),
        )
        self.block3 = nn.Sequential(
            InvertedResidual(c3, c3, stride=(1, 1), expand=2, se=True),
            InvertedResidual(c3, c4, stride=(1, 2), expand=2, se=True),
        )
        self.block4 = nn.Sequential(
            InvertedResidual(c4, c4, stride=(1, 1), expand=2, se=True),
            InvertedResidual(c4, c5, stride=(1, 1), expand=2, se=True),
        )
        self.pool = StatPool(dims=(2, 3))
        feat_dim = c5 * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.head(x)
        return x

    def stream_scores(self, x, window_frames: int = 128, hop_frames: int = 64):
        b, c, f, t = x.shape
        scores = []
        for s in range(0, max(1, t - window_frames + 1), hop_frames):
            e = s + window_frames
            xw = x[:, :, :, s:e]
            y = self.forward(xw)
            scores.append(torch.softmax(y, dim=1))
        if len(scores) == 0:
            y = self.forward(x)
            scores.append(torch.softmax(y, dim=1))
        return torch.stack(scores, dim=1)


def build_model(n_mels: int = 64, base_ch: int = 16, num_classes: int = 2):
    return LightweightDeepfakeDetector(n_mels=n_mels, base_ch=base_ch, num_classes=num_classes)

