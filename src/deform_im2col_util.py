import os
import torch
import torch.nn.functional as F

def image_meshgrid_from(x):
    # input: b,c,h,w
    # output: b,c,h,2
    shape = x.shape  # assume b,c,h,w
    _y, _x = torch.meshgrid(torch.arange(shape[2]), torch.arange(shape[3]))
    grid = torch.stack([_x, _y], dim=-1) #1,h,w,2
    return torch.stack([grid] * shape[0], dim=0).type(x.type()).to(x.device)


def normalize_meshgrid(grid):
    # normalize wrt to image size
    # input: b,h,w,2
    # output: b,h,w,2 (range = [-1,1])
    grid_new = torch.zeros_like(grid)
    b, h, w, _ = grid.shape
    grid_new[..., 0] = grid[..., 0] / (w - 1) * 2 - 1
    grid_new[..., 1] = grid[..., 1] / (h - 1) * 2 - 1
    return grid_new

def deform_im2col(im, offset, kernel_size=3):
    # Faster on gpu, slower on CPU
    # input: b,c,h,w
    # output: b,N*c,h*w
    with torch.no_grad():
        grid = image_meshgrid_from(im) #获得网格坐标 1,h,w,2
        b, c, h, w = im.shape

    N = kernel_size * kernel_size #25*25

    grid_ = torch.zeros(b * N, h, w, 2,  device=im.device).contiguous()
    im_ = im.repeat(N, 1, 1, 1)

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            grid_[(dy * kernel_size + dx) * b:(dy * kernel_size + dx + 1) * b] =\
                grid + offset + torch.tensor([dx - kernel_size // 2, dy - kernel_size // 2])[None, None, None, :].float().to(im.device)
            #第一轮循环计算h*w上每个点和他们相对位置上（-12，-12）的点的相关性
            #第二轮循环计算h*w上每个点和他们相对位置上（-11，-12）的点的相关性
            #以此类推直到计算与（12，12）的点的相关性，一共循环625轮
            #grid末尾两位从（0，0）到（227，119）torch.tensor末尾两位从（-12，-12）到（12，12）
            #一轮当中就是（0，0）+（-12，-12）+offset，（0，1）+（-12，-12）+offset.。。。。。（227，119）+（-12，-12）+offset
            #grid_[]大小为1，120，228，2 grid_整体大小为625，120，228，2

    out = F.grid_sample(im_.contiguous(), normalize_meshgrid(grid_).contiguous(), align_corners=True)
    out = out.reshape(N, b, c, h * w).permute(1,2,0,3)

    return out.reshape(b, kernel_size * kernel_size * c, h * w)
