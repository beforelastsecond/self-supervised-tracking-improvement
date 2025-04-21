import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, in_dim=256):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward_sing(self, x, mask_ratio=0.3, drop_out=False):
        """
        输入大小为b,c,h,w,返回尺寸相同(input+attention)
        attention尺寸为b,h*w,h*w

        """
        m_batchsize, C, height, width = x.size()  # 获得b,c,h,w

        proj_query = self.query(x).reshape(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).reshape(m_batchsize, -1, width * height)  # 获得q,k矩阵
        energy = proj_query.bmm(proj_key)  # q,k相乘

        # 做dropout
        if drop_out:
            m_r = torch.ones_like(energy) * mask_ratio
            energy = energy + torch.bernoulli(m_r) * -1e-12

        ###
        attention = self.softmax(energy)

        proj_value = self.value(x).reshape(m_batchsize, -1, width * height)  # 获得v矩阵

        out = proj_value.bmm(attention.permute(0, 2, 1))  # v和attention相乘
        out = out.reshape(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

    def forward(self, x):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = self.forward_sing(x.flatten(0, 1)).unflatten(0, (B, T))
            return x
        else:
            return self.forward_sing(x)


class MutiheadSelfAttention(nn.Module):  # 多头自注意力，posmebed挪到这里，drop默认为false，输入尺寸默认（b，c=256，h=64，w=64）

    def __init__(self, h=64, w=64, in_dim=256, head=4):
        super(MutiheadSelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.head = head

        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        #self.position = nn.Parameter(torch.zeros(1, in_dim,h, w)) #叠加位置

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward_sing(self, x, drop_out=False, mask_ratio=0.3):
        """
        输入大小为b,c,h,w,返回尺寸相同(input+attention)
        attention尺寸为b,h*w,h*w

        """
        m_batchsize, C, height, width = x.size()  # 获得b,c,h,w
        #position=F.interpolate(self.position, size=(height, width), mode='bicubic')
        #x = x + position #加入位置编码

        proj_query = self.query(x).view(m_batchsize, -1, self.head, width * height // self.head) \
            .permute(0, 2, 1, 3) # 获得q矩阵
        proj_key = self.key(x).view(m_batchsize, -1, self.head, width * height // self.head) \
            .permute(0, 2, 1, 3)  # 获得k矩阵
        proj_value = self.value(x).reshape(m_batchsize, -1, self.head, width * height // self.head) \
            .permute(0, 2, 1, 3)  # 获得v矩阵

        energy = torch.matmul(proj_query, proj_key.permute(0, 1, 3, 2))  # q,k相乘

        # 做dropout
        if drop_out:
            m_r = torch.ones_like(energy) * mask_ratio
            energy = energy + torch.bernoulli(m_r) * -1e-12

        ###
        attention = self.softmax(energy)

        out = torch.matmul(attention, proj_value)  # v和attention相乘
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

    def forward(self, x):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = self.forward_sing(x.flatten(0, 1)).unflatten(0, (B, T))
            return x
        else:
            return self.forward_sing(x)


if __name__ == '__main__':
    input = torch.randn(1, 256, 64, 64)
    # print(input)
    frame_in_dim = input.shape[1]  # 获得c
    self_attention = MutiheadSelfAttention(in_dim=frame_in_dim)
    # print(self_attention)
    output = self_attention(input)
    print(output - input)
    print(output.shape)
