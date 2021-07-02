import efficientnet_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

__all__ = ["get_n_params", "efficient_b0", "res_net50", "bot_net50_l1", "bot_net50_l2", "doge_net18_64x64",
           "doge_net50_64x64", "doge_net50_32x32", "doge_net18_32x32", "doge_net50_cifar"]


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


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


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
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


class DogeNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(DogeNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.ModuleList()
            self.conv2.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False))
            self.conv2.append(SE(planes, planes // 2))
            self.conv2 = nn.Sequential(*self.conv2)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
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


# reference
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class BotNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=15, resolution=(224, 224), heads=4,
                 layer3: str = "CNN"):
        super(BotNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # for ImageNet
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        if layer3 == "CNN":
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        elif layer3 == "Transformer":
            self.layer3 = self._make_layer(block, 256, num_blocks[3], stride=2, heads=heads, mhsa=True)
        else:
            raise NotImplementedError

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),  # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * block.expansion, num_classes),
        )

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)  # for ImageNet

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class DogeNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=15, resolution=(224, 224), heads=4):
        super(DogeNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
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
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
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


def efficient_b0(num_classes=15, **kwargs):
    return efficientnet_pytorch.EfficientNet.from_name("efficientnet-b0")  # 引用的最好的纯卷积神经网络


def res_net50(num_classes=15, **kwargs):
    return models.resnet50(num_classes=num_classes)  # 原始的resnet50，未加入transformer


def bot_net50_l1(num_classes=15, resolution=(224, 224), heads=4, **kwargs):
    return BotNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes,
                  resolution=resolution, heads=heads, layer3="CNN")  # resnet50加入一层transformer


def bot_net50_l2(num_classes=15, resolution=(224, 224), heads=4, **kwargs):
    return BotNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes,
                  resolution=resolution, heads=heads, layer3="Transformer")  # resnet50加入两层transformer


def doge_net18_64x64(num_classes=15, resolution=(64, 64), heads=4, **kwargs):
    return DogeNet(DogeNeck, [2, 3, 1, 2], num_classes=num_classes, resolution=resolution, heads=heads)


def doge_net50_64x64(num_classes=15, resolution=(64, 64), heads=4, **kwargs):
    return DogeNet(DogeNeck, [6, 6, 2, 2], num_classes=num_classes, resolution=resolution, heads=heads)


def doge_net18_32x32(num_classes=15, resolution=(32, 32), heads=4, **kwargs):
    return DogeNet(DogeNeck, [2, 3, 1, 2], num_classes=num_classes, resolution=resolution, heads=heads)


def doge_net50_32x32(num_classes=15, resolution=(32, 32), heads=4, **kwargs):
    return DogeNet(DogeNeck, [6, 6, 2, 2], num_classes=num_classes, resolution=resolution, heads=heads)


def doge_net50_cifar(num_classes=15, resolution=(32, 32), heads=4, **kwargs):
    return BotNet(BottleNeck, [6, 6, 2, 2], num_classes=num_classes,
                  resolution=resolution, heads=heads, layer3="Transformer")  # resnet50加入两层transformer


def main():
    x = torch.randn([2, 3, 32, 32])
    model = doge_net50_32x32(resolution=tuple(x.shape[2:]), heads=8)  # 18857295
    # model = doge_net50_64x64(resolution=tuple(x.shape[2:]), heads=8)  # 4178255
    # model = efficient_b0()
    # model = efficientnet_pytorch.EfficientNet.from_name("efficientnet-b0")

    print(model(x).size())
    print(get_n_params(model))

    # 打印网络结构
    summary(model, input_size=[(3, 32, 32)], batch_size=1, device="cpu")


if __name__ == '__main__':
    main()
