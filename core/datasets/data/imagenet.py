import os
from torchvision import datasets
from torchvision.transforms import transforms
from core.datasets.transforms.custom_transform import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def classify_train_dataset(data_dir, transform=TinyImageNetTrainTransform, **kwargs):
    train_dir = os.path.join(data_dir, 'train')
    return datasets.ImageNet(train_dir, transform=transform, download="False", **kwargs)


# val dataset example for tiny-image-net
def classify_val_dataset(data_dir, transform=TinyImageNetvalidationTransform, **kwargs):
    val_dir = os.path.join(data_dir, 'val')
    return datasets.ImageNet(val_dir, transform=transform, download="False", **kwargs)


# test dataset example for tiny-image-net
def classify_test_dataset(data_dir, transform=TinyImageNetTestTransform, **kwargs):
    test_dir = os.path.join(data_dir, 'test')
    return datasets.ImageNet(test_dir, transform=transform, download="False", **kwargs)
