import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim=256):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query(x).reshape(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).reshape(batch_size, -1, width * height)
        energy = proj_query.bmm(proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).reshape(batch_size, -1, width * height)
        out = proj_value.bmm(attention.permute(0, 2, 1))
        out = out.reshape(batch_size, C, height, width)
        out = self.gamma * out + x
        return out
