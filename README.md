# self-supervised-tracking-improvement

# 自监督目标追踪改进工作


## 项目结构

- `src/`：包含核心代码，涵盖模型实现、数据处理及训练脚本。
- `data/`：包括数据集及相关预处理脚本。



## 安装与使用

### 依赖

Pytorch == 1.8.4 & torchvision == 0.9.0 & Spatial-correlation-samplar== 0.3.0
- 其他相关依赖将在后续整理后提供。

### 训练
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222   --lr 1e-3


### 推理
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py --resume checkpoint.pt 

本项目的实验结果将会在整理后上传，用户可以在之后查看并执行相关脚本。详细的使用说明将随实验结果一起提供。
 


