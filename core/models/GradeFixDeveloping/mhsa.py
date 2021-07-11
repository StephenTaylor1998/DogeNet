import torch
from torch import nn


class MHSAX(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, position_embedding=True):
        super(MHSAX, self).__init__()
        self.heads = heads
        self.position_embedding = position_embedding
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=3, groups=n_dims, padding=1, padding_mode="replicate")
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1, groups=n_dims)
        if position_embedding:
            self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
            self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        if self.position_embedding:
            content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
            content_position = torch.matmul(content_position, q)
            energy = content_content + content_position
        else:
            energy = content_content

        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out


class DSA(nn.Module):
    def __init__(self, n_dims, view_size=3, attention_mode="A"):
        super(DSA, self).__init__()
        self.query = nn.Conv2d(n_dims, n_dims, 1)
        padding = (view_size - 1) // 2
        self.key = nn.Conv2d(n_dims, n_dims, view_size, groups=n_dims, padding=padding, padding_mode="replicate")
        self.value = nn.Conv2d(n_dims, n_dims, 1, groups=n_dims)
        self.attention_mode = attention_mode

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        if self.attention_mode == "A":
            # model "A"
            # attention = q + k
            # out = v * attention

            # testing
            out = v * q * k

        else:
            attention = q * k
            out = v + attention

        return out


if __name__ == '__main__':
    from core.models.GradeFixDeveloping.operator import FixSigmoid
    inp = torch.randn((1, 16, 64, 64))
    model = DSA(16)
    act = FixSigmoid()
    out = model(inp)
    print(out.shape)
    print(out.max())
    print(out.min())
    out = act(out)
    print(out.shape)
    print(out.max())
    print(out.min())
