import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.GradeFixDeveloping.mhsa import MHSAX, DSA

__all__ = ["shibax26", "shibax50", "dogex26", "dogex50",
           "shiba26", "shiba50", ]


class SE(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = F.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class DogeNeckX(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None, position_embedding=True):
        super(DogeNeckX, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.ModuleList()
            self.conv2.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False))
            self.conv2.append(SE(planes, planes // 2))
            self.conv2 = nn.Sequential(*self.conv2)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSAX(
                planes, width=int(resolution[0]), height=int(resolution[1]),
                heads=heads, position_embedding=position_embedding
            ))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ShibaNeckX(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, mhsa=False, attention_mode="A", **kwargs):
        super(ShibaNeckX, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.ModuleList()
            self.conv2.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False))
            self.conv2.append(SE(planes, planes // 2))
            self.conv2 = nn.Sequential(*self.conv2)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(DSA(planes, view_size=3, attention_mode=attention_mode))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HalfAttNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=15, resolution=(224, 224), heads=4, in_channel=3,
                 position_embedding=True):
        super(HalfAttNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)
        self.position_embedding = position_embedding

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if self.conv1.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 48, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 96, num_blocks[2], stride=2, heads=heads, mhsa=True)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=1, heads=heads, mhsa=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * block.expansion, num_classes),
        )

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            # in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None, position_embedding=True
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, heads=heads, mhsa=mhsa,
                                resolution=self.resolution, position_embedding=self.position_embedding))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class FullAttNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=15, resolution=(224, 224), heads=4, in_channel=3,
                 position_embedding=True):
        super(FullAttNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)
        self.position_embedding = position_embedding

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if self.conv1.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 48, num_blocks[1], stride=2, heads=heads, mhsa=True)
        self.layer3 = self._make_layer(block, 96, num_blocks[2], stride=2, heads=heads, mhsa=True)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=1, heads=heads, mhsa=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * block.expansion, num_classes),
        )

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            # in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None, position_embedding=True
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, heads=heads, mhsa=mhsa,
                                resolution=self.resolution, position_embedding=self.position_embedding))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def shiba26(num_classes=10, args=None, heads=4, **kwargs):
    in_shape = args.in_shape
    return FullAttNet(ShibaNeckX, [2, 2, 2, 2], num_classes=num_classes,
                      resolution=in_shape[1:], heads=heads, in_channel=in_shape[0])


def shiba50(num_classes=10, args=None, heads=4, **kwargs):
    in_shape = args.in_shape
    return FullAttNet(ShibaNeckX, [6, 6, 2, 2], num_classes=num_classes,
                      resolution=in_shape[1:], heads=heads, in_channel=in_shape[0])


def shibax26(num_classes=10, args=None, heads=4, **kwargs):
    in_shape = args.in_shape
    return HalfAttNet(ShibaNeckX, [2, 3, 1, 2], num_classes=num_classes,
                      resolution=in_shape[1:], heads=heads, in_channel=in_shape[0])


def shibax50(num_classes=10, args=None, heads=4, **kwargs):
    in_shape = args.in_shape
    return HalfAttNet(ShibaNeckX, [6, 6, 2, 2], num_classes=num_classes,
                      resolution=in_shape[1:], heads=heads, in_channel=in_shape[0])


def dogex26(num_classes=10, args=None, heads=4, **kwargs):
    in_shape = args.in_shape
    return HalfAttNet(DogeNeckX, [2, 3, 1, 2], num_classes=num_classes,
                      resolution=in_shape[1:], heads=heads, in_channel=in_shape[0])


def dogex50(num_classes=10, args=None, heads=4, **kwargs):
    in_shape = args.in_shape
    return HalfAttNet(DogeNeckX, [6, 6, 2, 2], num_classes=num_classes,
                      resolution=in_shape[1:], heads=heads, in_channel=in_shape[0])


if __name__ == '__main__':
    from core.models import get_n_params
    from torchsummary import summary
    from core.utils.argparse import arg_parse
    from fvcore.nn import flop_count, FlopCountAnalysis
    from core.models import doge_net26, doge_net50, b0

    args = arg_parse().parse_args()
    args.in_shape = (3, 224, 224)
    x = torch.randn([1, 3, 224, 224])
    # model = shibax26(args=args, heads=4)   # 0.824546 M   0.452469888 G
    # model = shibax50(args=args, heads=4)   # 1.111706 M   0.795923072 G
    # model = dogex26(args=args, heads=4)    # 0.837090 M   0.658994304 G
    # model = dogex50(args=args, heads=4)    # 1.126938 M   1.013511296 G
    model = b0()  # 5.288548 M   0.421872480 G
    # model = doge_net26(args=args, heads=4) # 0.917538 M   0.685035648 G
    # model = doge_net50(args=args, heads=4)  # 0.917538 M   0.685035648 G
    # model = shiba26(args=args, heads=4)    # 0.746298 M   0.379743360 G
    # model = shiba50(args=args, heads=4)  # 0.746298 M   0.581701760 G

    print(model(x).size())
    print(get_n_params(model))
    # 打印网络结构
    summary(model, input_size=[(3, 224, 224)], batch_size=1, device="cpu")
    flops = FlopCountAnalysis(model, inputs=(x,))
    print(flops.total())
