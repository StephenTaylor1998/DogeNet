import os
from torchvision import datasets
from torchvision.transforms import transforms
from core.datasets.transforms.custom_transform import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def classify_dataset(data_dir, transform, not_strict=False):
    if not os.path.exists(data_dir) and not_strict:
        print("path ==> '%s' is not found" % data_dir)
        return

    return datasets.ImageFolder(data_dir, transform)


# train dataset example for image-net
def classify_train_dataset(data_dir, transform=ImageNetTrainTransform):
    train_dir = os.path.join(data_dir, 'train')
    return datasets.ImageFolder(train_dir, transform)


# val dataset example for image-net
def classify_val_dataset(data_dir, transform=ImageNetValidationTransform):
    val_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(val_dir, transform)


# test dataset example for image-net
def classify_test_dataset(data_dir, transform=ImageNetTestTransform):
    test_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(test_dir, transform)


# # train dataset example for tiny-image-net
# def classify_train_dataset(data_dir, transform=TinyImageNetTrainTransform, **kwargs):
#     train_dir = os.path.join(data_dir, 'train')
#     return datasets.ImageFolder(train_dir, transform, **kwargs)
#
#
# # val dataset example for tiny-image-net
# def classify_val_dataset(data_dir, transform=TinyImageNetvalidationTransform, **kwargs):
#     val_dir = os.path.join(data_dir, 'val')
#     return datasets.ImageFolder(val_dir, transform, **kwargs)
#
#
# # test dataset example for tiny-image-net
# def classify_test_dataset(data_dir, transform=TinyImageNetTestTransform, **kwargs):
#     # test_dir = os.path.join(data_dir, 'test')
#     test_dir = os.path.join(data_dir, 'val')
#     return datasets.ImageFolder(test_dir, transform, **kwargs)
