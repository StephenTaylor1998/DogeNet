from torchvision.transforms import transforms

__all__ = ['ImageNetNormalize',
           'ImageNetTrainTransform',
           'ImageNetValidationTransform',
           'ImageNetTestTransform',
           'TinyImageNetTrainTransform',
           'TinyImageNetvalidationTransform',
           'TinyImageNetTestTransform',
           'MNISTTrainTransform',
           'MNISTValidationTransform',
           'MNISTTestTransform']

ImageNetNormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

# iamgenet examples
ImageNetTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetValidationTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetTestTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.Pad(57, padding_mode="symmetric"),
    transforms.RandomRotation((90, 90)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])

# tiny-imagenet examples
TinyImageNetTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    # you may remove this line, it's a test
    transforms.RandomVerticalFlip(),
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    transforms.ToTensor(),
    ImageNetNormalize,
])

TinyImageNetvalidationTransform = transforms.Compose([
    transforms.ToTensor(),
    ImageNetNormalize,
])

TinyImageNetTestTransform = transforms.Compose([
    transforms.ToTensor(),
    ImageNetNormalize,
])


# tiny-imagenet examples
MNISTTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.RandomHorizontalFlip(),
    # you may remove this line, it's a test
    transforms.RandomVerticalFlip(),
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.225]),
])

MNISTValidationTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.225]),
])

MNISTTestTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.225]),
])
