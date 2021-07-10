from efficientnet_pytorch import EfficientNet

__all__ = ['__down_load_weight__', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7',
           'b0_n_channel', 'b1_c1', 'b2_c1', 'b3_c1', 'b4_c1', 'b5_c1', 'b6_c1', 'b7_c1']


def __down_load_weight__():
    _b0 = b0()
    _b1 = b1()
    _b2 = b2()
    _b3 = b3()
    _b4 = b4()
    _b5 = b5()
    _b6 = b6()
    _b7 = b7()
    return [_b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7]


def b0(pretrained=False, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b0', num_classes=num_classes, **kwargs)


def b1(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b1', num_classes=num_classes, **kwargs)


def b2(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b2', num_classes=num_classes, **kwargs)


def b3(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b3', num_classes=num_classes, **kwargs)


def b4(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b4', num_classes=num_classes, **kwargs)


def b5(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b5', num_classes=num_classes, **kwargs)


def b6(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b6', num_classes=num_classes, **kwargs)


def b7(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b7', num_classes=num_classes, **kwargs)


def b0_n_channel(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, in_channels=1, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b0', num_classes=num_classes, in_channels=1, **kwargs)


def b1_c1(pretrained=True, num_classes=1000, in_channels=1, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(
            'efficientnet-b1', num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b1', num_classes=num_classes, in_channels=in_channels, **kwargs)


def b2_c1(pretrained=True, num_classes=1000, in_channels=1, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(
            'efficientnet-b2', num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b2', num_classes=num_classes, in_channels=in_channels, **kwargs)


def b3_c1(pretrained=True, num_classes=1000, in_channels=1, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(
            'efficientnet-b3', num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b3', num_classes=num_classes, in_channels=in_channels, **kwargs)


def b4_c1(pretrained=True, num_classes=1000, in_channels=1, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(
            'efficientnet-b4', num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b4', num_classes=num_classes, in_channels=in_channels, **kwargs)


def b5_c1(pretrained=True, num_classes=1000, in_channels=1, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(
            'efficientnet-b5', num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b5', num_classes=num_classes, in_channels=in_channels, **kwargs)


def b6_c1(pretrained=True, num_classes=1000, in_channels=1, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(
            'efficientnet-b6', num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b6', num_classes=num_classes, in_channels=in_channels, **kwargs)


def b7_c1(pretrained=True, num_classes=1000, in_channels=1, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(
            'efficientnet-b7', num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        return EfficientNet.from_name('efficientnet-b7', num_classes=num_classes, in_channels=in_channels, **kwargs)


if __name__ == '__main__':
    print(__all__)
    __down_load_weight__()
