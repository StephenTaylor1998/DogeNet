from torchvision import datasets
from torchvision.transforms import transforms

cifar_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])

cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    cifar_normalize,
])

cifar_transform_test = transforms.Compose([
    transforms.ToTensor(),
    cifar_normalize,
])


def classify_train_dataset(data_dir, transform=cifar_transform_train, **kwargs):
    return datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True, **kwargs)


# val dataset example for tiny-image-net
def classify_val_dataset(data_dir, transform=cifar_transform_test, **kwargs):
    return datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True, **kwargs)


# test dataset example for tiny-image-net
def classify_test_dataset(data_dir, transform=cifar_transform_test, **kwargs):
    return datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True, **kwargs)
