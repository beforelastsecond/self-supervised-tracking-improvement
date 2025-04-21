import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from .attention import SelfAttention
from .deform_im2col_util import deform_im2col
from spatial_correlation_sampler import SpatialCorrelationSampler


class MASTColorizer(nn.Module):
    def __init__(self, args, is_momen=False):
        super(MASTColorizer, self).__init__()
        self.args = args
        self.is_momen = is_momen  # 用来区分是动量模型还是语义模型

        # ResNet 作为特征提取器
        self.feature_extraction = resnet18()
        self.post_convolution = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.sa = SelfAttention(in_dim=256)  # 自注意力机制

        # Colorizer 初始化（无论是语义还是动量都共享）
        self.colorizer = Colorizer(
            D=4, R=6, C=7, bsize=args.bsize, factor=args.factor,
            is_training=args.training, compact=args.compact, semantic=args.semantic
        )

        # 动量更新（用于动量模型）
        if is_momen:
            self.m = args.enc_mo
            self.feats_queue = torch.randn(args.bsize * args.factor, 128, 32, 32)
            self.refs_queue = torch.randint(0, 10000, (args.bsize * args.factor,))
            self.flag = torch.zeros(1)

    def forward(self, rgb_r, quantized_r, rgb_t, ref_index=None, current_ind=None, refs=None, feats_r_semantic=None,
                feats_t_semantic=None):
        # 提取特征
        feats_r = [self.post_convolution(self.sa(self.feature_extraction(rgb))) for rgb in rgb_r]
        feats_t = self.post_convolution(self.sa(self.feature_extraction(rgb_t)))

        # 处理语义相关功能
        if self.args.semantic:
            feats_r_semantic = [self.conv_semantic(self.sa(self.feature_extraction(rgb))) for rgb in rgb_r]
            feats_t_semantic = self.conv_semantic(self.sa(self.feature_extraction(rgb_t)))
            quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind, feats_r_semantic,
                                         feats_t_semantic)
        else:
            quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)

        return quantized_t

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        # 动量更新逻辑
        pass

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feats_r, refs):
        # gather before updating queue
        feats_r = concat_all_gather(feats_r)
        refs = concat_all_gather(refs)

        colorizer = self.colorizer
        replace_flag = int(colorizer.flag[0]) % colorizer.factor
        colorizer.feats_queue[replace_flag * colorizer.bsize:(replace_flag + 1) * colorizer.bsize] = feats_r
        colorizer.refs_queue[replace_flag * colorizer.bsize:(replace_flag + 1) * colorizer.bsize] = refs
        colorizer.flag[0] += 1

    def prep(self, image, HW):
        _, c, _, _ = image.size()
        x = image.float()[:, :, ::self.D, ::self.D]  # 下采样
        if c == 1 and not self.training:
            x = one_hot(x.long(), self.C)
        return x

    def prep2(self, image, h, w):
        return F.interpolate(image, size=(h, w), mode='bilinear')


class Colorizer(nn.Module):
    def __init__(self, D=4, R=6, C=32, bsize=12, factor=3, is_training=False, compact=False, semantic=False):
        super(Colorizer, self).__init__()
        self.D = D
        self.R = R  # 窗口大小
        self.C = C

        self.P = self.R * 2 + 1
        self.N = self.P * self.P
        self.count = 0

        self.memory_patch_R = self.R
        self.memory_patch_P = self.memory_patch_R * 2 + 1
        self.memory_patch_N = self.memory_patch_P * self.memory_patch_P

        self.correlation_sampler_dilated = [
            SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.memory_patch_P,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=dirate) for dirate in range(2, 6)
        ]

        self.correlation_sampler_pixel = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)

        if semantic:
            self.correlation_sampler_region = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.R + 1,
                stride=1,
                padding=0,
                dilation=1)

        self.bsize = bsize
        self.factor = factor
        self.is_training = is_training
        self.compact = compact
        self.semantic = semantic

    def forward(self, feats_r, feats_t, quantized_r, ref_index, current_ind, feats_r_semantic=None,
                feats_t_semantic=None):
        nref = len(feats_r)
        nsearch = len([x for x in ref_index if current_ind - x > 15])

        dirates = [min(4, (current_ind - x) // 15 + 1) for x in ref_index if current_ind - x > 15]
        b, c, h, w = feats_t.size()
        N = self.P * self.P

        corrs_pixel = []
        corrs = []
        offset0 = []
        for searching_index in range(nsearch):
            samplerindex = dirates[searching_index] - 2
            coarse_search_correlation = self.correlation_sampler_dilated[samplerindex](feats_t, feats_r[
                searching_index])  # b, p, p, h, w
            coarse_search_correlation = coarse_search_correlation.reshape(b, self.memory_patch_N, h * w)
            coarse_search_correlation = F.softmax(coarse_search_correlation, dim=1)
            coarse_search_correlation = coarse_search_correlation.reshape(b, self.memory_patch_P, self.memory_patch_P,
                                                                          h, w, 1)
            _y, _x = torch.meshgrid(torch.arange(-self.memory_patch_R, self.memory_patch_R + 1),
                                    torch.arange(-self.memory_patch_R, self.memory_patch_R + 1))
            grid = torch.stack([_x, _y], dim=-1).unsqueeze(-2).unsqueeze(-2).reshape(1, self.memory_patch_P,
                                                                                     self.memory_patch_P, 1, 1,
                                                                                     2).float().to(
                coarse_search_correlation.device)
            offset0.append((coarse_search_correlation * grid).sum(1).sum(1) * dirates[searching_index])
            col_0 = deform_im2col(feats_r[searching_index], offset0[-1], kernel_size=self.P)  # b,c*N,h*w
            col_0 = col_0.reshape(b, c, N, h, w)
            corr = (feats_t.unsqueeze(2) * col_0).sum(1)  # (b, N, h, w)
            corr = corr.reshape([b, self.P * self.P, h * w])
            corrs_pixel.append(corr)

        # 合并像素级的相关性并计算
        corr_pixel = torch.cat(corrs_pixel, 1)
        corr_pixel = F.softmax(corr_pixel, dim=1)
        corr_pixel = corr_pixel.unsqueeze(1)

        if self.semantic:
            corrs = torch.cat(corrs, 1)
            corrs = F.softmax(corrs, dim=1)
            corrs = corrs.unsqueeze(1)

        return corr_pixel

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
