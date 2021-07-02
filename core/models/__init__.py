# ------------------------------------------------ classify ------------------------------------------------ #
# import models from torchvision
from torchvision.models import *
# import models from efficientnet
from .efficientnet import b0, b1, b2, b3, b4, b5, b6, b7
from .efficientnet import b0_n_channel, b1_c1, b2_c1, b3_c1, b4_c1, b5_c1, b6_c1, b7_c1

# import models from resnet_m
from core.models.other_models.resnet_m import resnet18_tiny, resnet50_tiny, resnet50_tiny_c1, resnet50_c1

# import models from other_models
from .other_models import resnet18_cifar, resnet34_cifar, resnet50_cifar, \
    resnext29_2x64d_cifar, resnext29_32x4d_cifar
from .other_models import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from .other_models import DLA, SimpleDLA, DPN26, DPN92, EfficientNetB0, GoogLeNet, LeNet
from .other_models import PNASNetA, PNASNetB, MobileNet, MobileNetV2
from .other_models import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet152, PreActResNet101
from .other_models import RegNetX_200MF, RegNetX_400MF, SENet18
# from .other_models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
# from .other_models import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
# from .other_models import ShuffleNetG2, ShuffleNetG3, ShuffleNetV2, ResNet101, ResNet152
# from .other_models import VGG11, VGG13, VGG16, VGG19
from .transformer import *
# ------------------------------------------------ classify ------------------------------------------------ #
