import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class M2(nn.Module):
    """
    out = (w1 * (|x|/x) * x^2) + (w2 * x) + b
    """

    def __init__(self):
        super(M2, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        signal = x / torch.abs(x)
        x = (x ** 2 * signal) * self.w1 + x * self.w2 + self.b
        return x


class M3(nn.Module):
    """
    out = (w1 * x^3) + (w3 * x) + b
    """

    def __init__(self):
        super(M3, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x = (x ** 3) * self.w1 + x * self.w2 + self.b
        return x


class Conv2dM3(nn.Module):
    """
    out = (w1 * x^3) + (w3 * x) + b
    """

    def __init__(self, n_dims, view_size=3, attention_mode="A"):
        super(Conv2dM3, self).__init__()
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias=True)
        padding = (view_size - 1) // 2
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=view_size, groups=n_dims, padding=padding,
                             padding_mode="replicate", bias=True)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        x = q * k * v / 1.618
        return x


class Binary(nn.Module):
    """
    x = min_value if x < min_value
    x = max_value if x > max_value
    min_value < out < max_value
    """

    def __init__(self, min_value: float = -1, max_value: float = 1):
        super(Binary, self).__init__()
        self.min_value = torch.tensor(min_value, dtype=torch.float32)
        self.max_value = torch.tensor(max_value, dtype=torch.float32)

    def forward(self, x):
        x = torch.where(x > self.max_value, self.max_value, x)
        x = torch.where(x < self.min_value, self.min_value, x)
        return x


class FixBinary(nn.Module):
    """
    x = min_value if x < min_value
    x = max_value if x > max_value
    min_value < out < max_value
    """

    def __init__(self, min_value: float = -1, max_value: float = 1,
                 trainable = True):
        super(FixBinary, self).__init__()
        if trainable:
            self.min_value = nn.Parameter(torch.tensor(min_value, dtype=torch.float32), requires_grad=True)
            self.max_value = nn.Parameter(torch.tensor(max_value, dtype=torch.float32), requires_grad=True)
        else:
            self.min_value = torch.tensor(min_value, dtype=torch.float32)
            self.max_value = torch.tensor(max_value, dtype=torch.float32)

    def forward(self, x):
        origin = x
        x = torch.where(x > self.max_value, self.max_value, x)
        x = torch.where(x < self.min_value, self.min_value, x)
        # re-parameter operator
        x = Variable(x.data - origin.data) + origin
        return x


class FixSigmoid(nn.Module):
    def __init__(self):
        super(FixSigmoid, self).__init__()

    def forward(self, x):
        origin = x
        x = F.sigmoid(x)
        x = Variable(x.data - origin.data) + origin
        return x


if __name__ == '__main__':
    inp = torch.randn((1, 3, 224, 224))
    print(inp.max())
    print(inp.min())
    layer = Conv2dM3(n_dims=3)
    out = layer(inp)
    # print(out)
    print(out.max())
    print(out.min())
    print(out.mean())
