import argparse
import os, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn
import logger
import numpy as np
from jhmdbset import JHMDBSet


def getmax(x):
    row, col = torch.argmax(torch.max(x, 1).values, 0), torch.argmax(torch.max(x, 0).values, 0)
    coord = torch.tensor([row, col])
    return coord


def process_pose(pred, topk=100):  # 【H,W,C】
    # generate the coordinates:
    flatlbls = pred.flatten(0, 1)
    topk = min(flatlbls.shape[0], topk)
    vals, ids = torch.topk(flatlbls, k=topk, dim=0)
    vals /= vals.sum(0)[None]
    xx, yy = ids % pred.shape[1], ids // pred.shape[1]

    current_coord = torch.stack([(xx * vals).sum(0), (yy * vals).sum(0)], dim=0)
    current_coord[:, flatlbls.sum(0) == 0] = -1

    return current_coord  # [2,15]


def draw_labelmap_np(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]  # 坐标缩小到四分之一
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def main():
    args.bsize = 12
    args.factor = 1
    args.training = False
    args.semantic = False
    args.pad_divisible = 4
    args.enc_mo = 0.999

    if args.usemomen:
        from models.mast_momen import MAST

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/jhmdb_benchmark.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))  # 保存args参数

    TrainData = JHMDBSet(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(TrainData, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    model = MAST(args).cuda()
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')
    model = nn.DataParallel(model).cuda()

    start_full_time = time.time()

    test(TrainImgLoader, model, log)
    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

    return


def test(dataloader, model, log):
    model.eval()
    torch.backends.cudnn.benchmark = True
    log.info("Start testing.")

    for idx, (imgs, annos) in enumerate(dataloader):
        print('===> Processing {:3n} video.'.format(idx))

        total_frame_num = len(imgs)  # 当前序列的png图像数量
        print('total_frame_num: ' + str(total_frame_num))
        imgs = [r.cuda() for r in imgs]
        annotations = [q.cuda() for q in annos]
        cur_b, cur_c, cur_h, cur_w = imgs[0].shape  # 记录尺寸
        padded_height, padded_width = cur_h, cur_w
        outputs = [annotations[0].contiguous()]  # 仅包含第一帧坐标[15,H,W]
        all_coord = np.zeros((2, 15, total_frame_num))  # 用于存储坐标

        for i in range(total_frame_num - 1):

            mem_gap = 2
            if args.ref == 0:
                ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(
                    filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
                ref_index = sorted(list(set(ref_index)))
            elif args.ref == 1:
                ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
            elif args.ref == 2:
                ref_index = [i]
            else:
                raise NotImplementedError

            rgb_0 = [imgs[ind] for ind in ref_index]
            rgb_1 = imgs[i + 1]  # 生成参考帧和目标帧
            anno_0 = [outputs[ind] for ind in ref_index]

            _, _, h, w = anno_0[0].size()  # [1,15,h,w]

            if i == 0:
                all_coord[:, :, 0] = process_pose(annotations[0][0].permute(1, 2, 0)).cpu().numpy()

            with torch.no_grad():
                _output = model(rgb_0, anno_0, rgb_1, ref_index, i + 1)  # 返回的尺寸[1,15,H/4,W/4]
                _output = F.interpolate(_output, (padded_height, padded_width), mode='bilinear')  # [1,15,H,W]
                all_coord[:, :, i + 1] = process_pose(_output[0].permute(1, 2, 0)).cpu().numpy()
                outputs.append(_output)

        info = '\t'.join(['idx of video: {:3n} '.format(idx)])

        log.info('[{},over] '.format(info))

        coordname = os.path.join(args.savepath, str(idx) + '.dat')
        all_coord.dump(coordname)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LIIR_jhmdb')

    # Data options
    parser.add_argument('--ref', type=int, default=0)

    parser.add_argument('--datapath', help='Data path for JHMDB', default='/dataset/dai212/')
    parser.add_argument('--savepath', type=str, default='results_jhmdb/',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--usemomen', action='store_true',
                        help='use momentum encoder')

    args = parser.parse_args()

    main()
