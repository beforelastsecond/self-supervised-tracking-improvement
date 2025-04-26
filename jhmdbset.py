import torch.utils.data as data
import os
import numpy as np
import cv2
import scipy.io as sio
import scipy.misc
import torch
from matplotlib import cm
import torchvision.transforms as transforms


def img_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.from_numpy(img).float()
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([50, 0, 0], [50, 127, 127])(img)

    return img


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def draw_labelmap_np(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]  #坐标缩小到四分之一
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


class JHMDBSet(data.Dataset):
    def __init__(self,filelist):

        self.sigma = 0.5
        self.file_name_list ='val_list.txt'
        self.filelist=filelist

        f = open(self.file_name_list, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = self.filelist+rows[1]  # 获取图像文件夹名称
            lblfile = self.filelist+rows[0]  # 获取关键点信息文件夹名称

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()

    def __getitem__(self, index):
        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]  # 图像和关键点信息序列
        imgs = []

        folder_contains = os.listdir(folder_path)  # 当前序列下每个文件

        pngcnt = 0
        for i in range(len(folder_contains)):
            if '.png' in folder_contains[i]:
                pngcnt = pngcnt + 1  # 进行计数，每有一个png就加1
        frame_num = pngcnt  # 当前序列图像总数

        for i in range(frame_num):
            img_path = folder_path + "/{:05d}.png".format(i + 1)
            img = load_image(img_path)
            imgs.append(img)

        H, W = imgs[0].size(1), imgs[0].size(2)  # 获取图像尺寸

        lbls_mat = sio.loadmat(label_path)  # 载入matlab文件,
        lbls_coord = lbls_mat['pos_img']  # 尺寸为[2,15,num_png],每幅图有15个点坐标
        lbls_coord = lbls_coord - 1

        lblsize = (H, W)  # 标签尺寸与图像一致
        #lblsize = (H//4, W//4) #缩小到四分之一做标签

        lbls = np.zeros((lbls_coord.shape[2], lblsize[0], lblsize[1], lbls_coord.shape[1]))  # [num_png,H,W,15]

        for i in range(lbls_coord.shape[2]):
            lbls_coord_now = lbls_coord[:, :, i]  # 获取当前图像点信息

            for j in range(lbls_coord.shape[1]):  # 在lbls上绘制每一个点
                if self.sigma > 0:
                    draw_labelmap_np(lbls[i, :, :, j], lbls_coord_now[:, j], self.sigma)
                else:
                    tx = int(lbls_coord_now[0, j])
                    ty = int(lbls_coord_now[1, j])
                    if tx < lblsize[1] and ty < lblsize[0] and tx >= 0 and ty >= 0:
                        lbls[i, ty, tx, j] = 1.0

        #lbls_tensor = torch.zeros(frame_num, lblsize[0], lblsize[1], lbls_coord.shape[1])  # [num_png,H,W,15]
        lbls_tensor = torch.zeros(frame_num, lblsize[0], lblsize[1], lbls_coord.shape[1])

        for i in range(frame_num):
            lbls_tensor[i] = torch.from_numpy(lbls[i])

        lbls_tensor = lbls_tensor.permute(0, 3, 1, 2)
        lbls_tensor = [lbls_tensor[i] for i in range(lbls_tensor.shape[0])]

        return imgs, lbls_tensor #返回两个list，图片和点坐标

    def __len__(self):
        return len(self.jpgfiles)


