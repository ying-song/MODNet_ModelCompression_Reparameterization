import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import BatchNorm2d

# from .backbones import SUPPORTED_BACKBONES
from src.models.backbones import SUPPORTED_BACKBONES


class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels

        self.bnorm_channels = max(1, int(1 * in_channels / 2))
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 低分辨率
# class lrRepConv2dIBNormRelu(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=1, dilation=1, groups=1,
#                  with_ibn=True, with_relu=True):
#         super().__init__()
#
#         # 统一使用same padding逻辑
#         if padding is None:
#             padding = (kernel_size - 1) // 2 * dilation
#
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.with_ibn = with_ibn
#         self.with_relu = with_relu
#         self.deploy = False
#
#         # 仅处理3x3卷积的重参数化
#         if kernel_size == 3:
#             # 主分支：3x3卷积（保留偏置）
#             self.rbr_dense = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=3,
#                 stride=stride, padding=padding, dilation=dilation,
#                 groups=groups, bias=True
#             )
#
#             # 1x1分支：等效3x3卷积核中心点
#             self.rbr_1x1 = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=1,
#                 stride=stride, padding=0, groups=groups, bias=True
#             )
#
#             # 身份映射分支（自动处理通道数变化）
#             self.rbr_identity = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=1,
#                 stride=stride, padding=0, groups=groups, bias=False
#             ) if (in_channels == out_channels and stride == 1) else None
#
#             self.rbr_identity = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=1,
#                 stride=1, padding=0, groups=groups, bias=False
#             ) if (in_channels == out_channels and stride == 1) else None
#
#             # 可学习缩放系数（使用更稳定的初始化）
#             self.scale_dense = nn.Parameter(torch.ones(out_channels, 1, 1))
#             self.scale_1x1 = nn.Parameter(torch.ones(out_channels, 1, 1))
#             self.scale_identity = nn.Parameter(torch.ones(out_channels, 1, 1)) if self.rbr_identity else None
#
#             # IBNorm层（记录必要参数）
#             if with_ibn:
#                 self.ibn = IBNorm(out_channels)
#                 self.eps = self.ibn.bnorm.eps
#                 self.momentum = self.ibn.bnorm.momentum
#             else:
#                 self.ibn = None
#
#             # 权重初始化
#             self._init_weights()
#
#         else:
#             # 非3x3卷积保持原结构
#             self.layers = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size,
#                           stride=stride, padding=padding, dilation=dilation,
#                           groups=groups, bias=True),
#                 IBNorm(out_channels) if with_ibn else nn.Identity(),
#                 nn.ReLU(inplace=True) if with_relu else nn.Identity()
#             )
#
#     def _init_weights(self):
#         # 保持与原模型一致的初始化
#         nn.init.kaiming_normal_(self.rbr_dense.weight, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.rbr_1x1.weight, mode='fan_out', nonlinearity='relu')
#
#         scale = 1 / (sum([1 for _ in [self.rbr_dense, self.rbr_1x1, self.rbr_identity] if _ is not None]) ** 0.5)
#         self.rbr_dense.weight.data.mul_(scale)
#         self.rbr_1x1.weight.data.mul_(scale)
#         if self.rbr_identity:
#             self.rbr_identity.weight.data.mul_(scale)
#
#         if self.rbr_identity is not None:
#             nn.init.ones_(self.rbr_identity.weight)
#
#         # 缩放参数初始化
#         nn.init.ones_(self.scale_dense)
#         nn.init.ones_(self.scale_1x1)
#         if self.scale_identity is not None:
#             nn.init.ones_(self.scale_identity)
#
#     def forward(self, x):
#         if self.kernel_size != 3 or self.deploy:
#             return self.layers(x)
#
#         # 训练阶段多分支计算
#         dense_out = self.rbr_dense(x) * self.scale_dense
#         x1x1 = self.rbr_1x1(x) * self.scale_1x1
#
#         # 自动对齐特征图尺寸
#         x1x1 = self._pad_to_match(x1x1, dense_out.shape)
#
#         # 身份映射分支
#         if self.rbr_identity is not None:
#             id_out = self.rbr_identity(x) * self.scale_identity
#             id_out = self._pad_to_match(id_out, dense_out.shape)
#         else:
#             id_out = 0
#
#         fused = dense_out + x1x1 + id_out
#
#         # 应用IBNorm和ReLU
#         if self.ibn is not None:
#             fused = self.ibn(fused)
#         if self.with_relu:
#             fused = F.relu(fused, inplace=True)
#
#         return fused
#
#     def _pad_to_match(self, tensor, target_shape):
#         """动态填充特征图到目标尺寸"""
#         _, _, h, w = target_shape
#         # return F.interpolate(tensor, size=(h, w), mode='nearest')
#         return F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)
#
#     def get_fused_kernel_bias(self):
#         """融合卷积核与IBNorm参数"""
#         # 基础卷积核融合
#         kernel = (
#                 self.rbr_dense.weight * self.scale_dense.view(-1, 1, 1, 1) +
#                 F.pad(self.rbr_1x1.weight * self.scale_1x1.view(-1, 1, 1, 1), [1, 1, 1, 1])
#         )
#
#         if self.rbr_identity is not None:
#             kernel += F.pad(self.rbr_identity.weight * self.scale_identity.view(-1, 1, 1, 1), [1, 1, 1, 1])
#
#         # 融合偏置
#         bias = (
#                 self.rbr_dense.bias * self.scale_dense.flatten() +
#                 self.rbr_1x1.bias * self.scale_1x1.flatten()
#         )
#
#         if self.rbr_identity is not None:
#             bias += self.rbr_identity.bias * self.scale_identity.flatten()
#
#         # 融合IBNorm参数(单独修改lr层）
#         if self.ibn is not None:
#             gamma = self.ibn.weight
#             beta = self.ibn.bias
#             running_var = self.ibn.running_var
#             eps = self.ibn.eps
#
#             scale = gamma / torch.sqrt(running_var + eps)
#             kernel *= scale.view(-1, 1, 1, 1)
#             bias = bias * scale + (beta - gamma * self.ibn.running_mean / torch.sqrt(running_var + eps))
#
#         return kernel, bias
#         # # 多层结果较好,但没有达到原始效果
#         # if self.ibn is not None:
#         #     bn_weight = self.ibn.bnorm.weight
#         #     bn_bias = self.ibn.bnorm.bias
#         #     bn_mean = self.ibn.bnorm.mean
#         #     bn_var = self.ibn.bnorm.running_var
#         #     eps = self.ibn.bnorm.eps
#         #
#         #     scale_bn = bn_weight / torch.sqrt(bn_var + self.ibn.bnorm.eps)
#         #     kernel[:, :self.ibn.bnorm_channels] *= scale_bn.view(-1, 1, 1, 1)
#         #     bias[:self.ibn.bnorm_channels] = (bias[:self.ibn.bnorm_channels] * scale_bn + bn_bias - scale_bn * bn_mean)
#         #
#         # return kernel, bias
#
#     def switch_to_deploy(self):
#         if self.kernel_size == 3 and not self.deploy:
#             # 生成融合参数
#             kernel, bias = self.get_fused_kernel_bias()
#
#             # 创建部署用卷积层
#             self.layers = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=self.rbr_dense.in_channels,
#                     out_channels=self.rbr_dense.out_channels,
#                     kernel_size=3,
#                     stride=self.stride,
#                     padding=self.padding,
#                     dilation=self.dilation,
#                     groups=self.groups,
#                     bias=True
#                 ),
#                 nn.ReLU(inplace=True) if self.with_relu else nn.Identity()
#             )
#
#             # 加载融合参数
#             self.layers[0].weight.data = kernel
#             self.layers[0].bias.data = bias
#
#             # 清理训练参数
#             del self.rbr_dense, self.rbr_1x1, self.rbr_identity
#             del self.scale_dense, self.scale_1x1, self.scale_identity
#             if hasattr(self, 'ibn'):
#                 del self.ibn
#
#             self.deploy = True


# 方法二：将IBNorm提到主分支
# class RepConv2dIBNormRelu(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, bias=True,
#                  with_ibn=True, with_relu=True, deploy=False):
#         super(RepConv2dIBNormRelu, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.with_ibn = with_ibn
#         self.with_relu = with_relu
#
#         # 仅对3x3非分组卷积进行重参数化
#         if kernel_size == 3 and groups == 1 and not deploy:
#             self.rbr_3x3 = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=3,
#                 stride=stride, padding=padding, bias=False)
#             self.rbr_1x1 = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=1,
#                 stride=stride, padding=0, bias=False)
#
#             # Identity分支条件
#             self.rbr_identity = None
#             if out_channels == in_channels and stride == 1:
#                 self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
#                 nn.init.constant_(self.rbr_identity.weight, 1.0)
#                 nn.init.constant_(self.rbr_identity.bias, 0)
#         else:
#             # 普通卷积路径
#             self.rbr_reparam = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=kernel_size,
#                 stride=stride, padding=padding, dilation=dilation,
#                 groups=groups, bias=bias)
#
#         # 标准化和激活
#         if with_ibn:
#             self.ibn = IBNorm(out_channels)
#         if with_relu:
#             self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         if hasattr(self, 'rbr_3x3'):
#             x3 = self.rbr_3x3(x)
#             x1 = self.rbr_1x1(x)
#             x_id = 0
#             if self.rbr_identity is not None:
#                 x_id = self.rbr_identity(x)
#             x = x3 + x1 + x_id
#         else:
#             x = self.rbr_reparam(x)
#
#         if hasattr(self, 'ibn'):
#             x = self.ibn(x)
#         if hasattr(self, 'relu'):
#             x = self.relu(x)
#         return x
#
#     def reparametrize(self):
#         if self.deploy or not hasattr(self, 'rbr_3x3'):
#             return
#
#         # 融合卷积和BN
#         kernel3x3, bias3x3 = self._fuse_bn(self.rbr_3x3)
#         kernel1x1, bias1x1 = self._fuse_bn(self.rbr_1x1)
#         kernelid, biasid = self._fuse_identity()
#
#         # 合并kernel
#         kernel = kernel3x3 + self._pad_1x1_to_3x3(kernel1x1) + kernelid
#         bias = bias3x3 + bias1x1 + biasid
#
#         # 创建重参数化卷积
#         self.rbr_reparam = nn.Conv2d(
#             in_channels=self.rbr_3x3.in_channels,
#             out_channels=self.rbr_3x3.out_channels,
#             kernel_size=3,
#             stride=self.rbr_3x3.stride,
#             padding=self.rbr_3x3.padding,
#             dilation=self.rbr_3x3.dilation,
#             groups=self.rbr_3x3.groups,
#             bias=True)
#         self.rbr_reparam.weight.data = kernel
#         self.rbr_reparam.bias.data = bias
#
#         # 删除训练用参数
#         del self.rbr_3x3, self.rbr_1x1
#         if self.rbr_identity:
#             del self.rbr_identity
#         self.deploy = True
#
#     def _fuse_bn(self, conv):
#         # 将conv+bn融合为带bias的conv
#         kernel = conv.weight
#         running_mean = 0
#         running_var = 1
#         gamma = 1
#         beta = 0
#
#         # 计算融合后的kernel和bias
#         fused_kernel = kernel * gamma.view(-1, 1, 1, 1)
#         fused_bias = beta - gamma * running_mean
#         return fused_kernel, fused_bias
#
#     def _fuse_identity(self):
#         if not self.rbr_identity:
#             return 0, 0
#
#         # 处理identity分支
#         input_dim = self.rbr_identity.num_features
#         kernel = torch.zeros((input_dim, input_dim, 3, 3), device=self.rbr_identity.weight.device)
#         for i in range(input_dim):
#             kernel[i, i, 1, 1] = 1.0
#         return kernel * self.rbr_identity.weight.view(-1, 1, 1, 1), self.rbr_identity.bias
#
#     def _pad_1x1_to_3x3(self, kernel):
#         return F.pad(kernel, [1, 1, 1, 1])

# # 方法一：
class SplitBatchNorm(nn.BatchNorm2d):
    """前一半通道可学习BN，后一半固定恒等变换"""

    def __init__(self, num_features, split):
        super().__init__(num_features)
        self.split = split

        # 初始化后半参数为固定值
        with torch.no_grad():
            self.weight[self.split:].fill_(1.0)  # 后半gamma固定为1
            self.bias[self.split:].fill_(0.0)  # 后半beta固定为0

        # 冻结参数
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)
        # 强制释放不需要的内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class hrRepConv2dIBNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True, deploy=False):
        super().__init__()
        assert kernel_size == 3 and groups == 1, "仅支持3x3非分组卷积"

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.with_ibn = with_ibn
        self.with_relu = with_relu
        self.deploy = deploy
        self._optimize_memory = False

        if not self.deploy:
            # 定义三个分支
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3,
                                     stride, padding, dilation, groups, bias)
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1,
                                     stride, 0, dilation, groups, bias)

            # Identity分支（固定参数的1x1卷积）
            self.id_conv = nn.Conv2d(in_channels, out_channels, 1,
                                     stride, 0, dilation, groups, bias)
            self._init_identity_conv()

            # 每个分支后接SplitBatchNorm
            split_point = out_channels // 2
            self.bn3x3 = SplitBatchNorm(out_channels, split_point)
            self.bn1x1 = SplitBatchNorm(out_channels, split_point)
            self.bn_id = SplitBatchNorm(out_channels, split_point)

        if self.deploy:
            self.fused_conv = nn.Conv2d(in_channels, out_channels, 3,
                                        stride, padding, dilation, groups, bias)

        # 后续处理模块
        if self.with_ibn:
            self.inorm = nn.InstanceNorm2d(out_channels // 2, affine=False)
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def _init_identity_conv(self):
        """初始化Identity卷积为恒等映射"""
        with torch.no_grad():
            if self.in_channels == self.out_channels:
                # 单位矩阵初始化
                self.id_conv.weight.data = torch.eye(self.out_channels).view(
                    self.out_channels, self.in_channels, 1, 1)
            else:
                # 通道不匹配时部分通道恒等
                min_dim = min(self.in_channels, self.out_channels)
                self.id_conv.weight.data[:min_dim, :min_dim, 0, 0] = torch.eye(min_dim)
            if self.id_conv.bias is not None:
                self.id_conv.bias.data.zero_()
        # 冻结参数
        self.id_conv.weight.requires_grad_(False)
        if self.id_conv.bias is not None:
            self.id_conv.bias.requires_grad_(False)

    def forward(self, x):
        if self.deploy:
            out = self.fused_conv(x)
        else:
            # 三分支前向计算
            x3 = self.bn3x3(self.conv3x3(x))
            x1 = self.bn1x1(self.conv1x1(x))
            x_id = self.bn_id(self.id_conv(x))

            out = x3.add_(x1).add_(x_id)
            del x3, x1, x_id

        # InstanceNorm处理后半通道
        if self.with_ibn:
            split = self.out_channels // 2
            front = out[:, :split]
            back = self.inorm(out[:, split:])
            out = torch.cat([front, back], dim=1)

        # ReLU激活
        if self.with_relu:
            out = self.relu(out)

        return out

    def _fuse_bn_tensor(self, conv, bn):
        """融合卷积与BN参数"""
        kernel = conv.weight
        bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

        scale_factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_kernel = (kernel * scale_factor.reshape(-1, 1, 1, 1))
        fused_bias = scale_factor * (bias - bn.running_mean) + bn.bias

        return fused_kernel, fused_bias

    def reparameterize(self):
        """三分支融合为单卷积"""
        if self.deploy or not hasattr(self, 'conv3x3'):
            return

        device = self.conv3x3.weight.device

        # 融合3x3分支
        kernel3, bias3 = self._fuse_bn_tensor(self.conv3x3, self.bn3x3)

        # 融合1x1分支（补零为3x3）
        kernel1 = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        conv1x1_fused = nn.Conv2d(
            self.in_channels, self.out_channels, 3,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=1, bias=True
        ).to(self.conv1x1.weight.device)
        conv1x1_fused.weight.data = kernel1
        conv1x1_fused.bias.data = self.conv1x1.bias if self.conv1x1.bias is not None else 0
        kernel1, bias1 = self._fuse_bn_tensor(conv1x1_fused, self.bn1x1)

        # 融合Identity分支（补零为3x3）
        kernel_id = F.pad(self.id_conv.weight, [1, 1, 1, 1])
        conv_id_fused = nn.Conv2d(
            self.in_channels, self.out_channels, 3,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=1, bias=True
        ).to(self.id_conv.weight.device)
        conv_id_fused.weight.data = kernel_id
        conv_id_fused.bias.data = self.id_conv.bias if self.id_conv.bias is not None else 0
        kernel_id, bias_id = self._fuse_bn_tensor(conv_id_fused, self.bn_id)

        # 合并三分支参数
        final_kernel = kernel3 + kernel1 + kernel_id
        final_bias = bias3 + bias1 + bias_id

        # 创建融合卷积
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, 3,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=1, bias=True
        ).to(device)

        self.fused_conv.load_state_dict({
            'weight': final_kernel,
            'bias': final_bias
        })

        self.__dict__['_modules']['conv3x3'] = None
        self.__dict__['_modules']['conv1x1'] = None
        self.__dict__['_modules']['id_conv'] = None
        self.__dict__['_modules']['bn3x3'] = None
        self.__dict__['_modules']['bn1x1'] = None
        self.__dict__['_modules']['bn_id'] = None

        self.deploy = True

    def switch_to_deploy(self):
        """切换到部署模式"""
        if not self.deploy:
            self.reparameterize()

    def enable_memory_optim(self, enable=True):
        self._optimize_memory = enable
        if enable:
            torch.backends.cudnn.benchmark = False
            torch.cuda.set_per_process_memory_fraction(0.5)


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf """

    def __init__(self, in_channels, out_channels=None, reduction=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._make_fc(self.in_channels, self.out_channels, reduction)

    def _make_fc(self, in_channels, out_channels, reduction):
        reduced_channels = max(1, in_channels // reduction)
        return nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Check if input channels match the expected channels
        if c != self.in_channels:
            raise ValueError(
                f"Input has {c} channels, expected {self.in_channels}. Please ensure your input size is correct.")

        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, self.out_channels, 1, 1)

        return x * w.expand_as(x)


# ------------------------------------------------------------------------------
#  MODNet Branches
# ------------------------------------------------------------------------------

# The rest of the code remains the same

# 原版
class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone, lr_channel, lr16x_out=None):
        super(LRBranch, self).__init__()

        first_se_channel = backbone.last_channel

        if lr16x_out is None:
            lr16x_out = lr_channel[1]

        self.backbone = backbone
        self.se_block = SEBlock(first_se_channel, first_se_channel, reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(first_se_channel, lr16x_out, 5, stride=1, padding=2)  # 1280,96
        self.conv_lr8x = Conv2dIBNormRelu(lr16x_out, lr_channel[0], 5, stride=1, padding=2)  # 96,32
        self.conv_lr = Conv2dIBNormRelu(lr_channel[0], 1, kernel_size=3, stride=2, padding=1, with_ibn=False,
                                             with_relu=False)

    def forward(self, img, inference=False):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]
        enc32x = self.se_block(enc32x)  # [1,1280,16,16]
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        lr8x = self.conv_lr8x(lr8x)  # [1,32,64,64]

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x]


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, hr_channel):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(hr_channel[0], hr_channels, 1, stride=1, padding=0)  # 16,32
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, kernel_size=3, stride=2, padding=1,
                                                groups=1, with_ibn=True, with_relu=True)  # 35,32

        # self.conv_enc2x = Conv2dIBNormRelu(19, hr_channels, 3, stride=2, padding=1)  # 35,32

        self.tohr_enc4x = Conv2dIBNormRelu(hr_channel[1], hr_channels, 1, stride=1, padding=0)  # 24,32
        # self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1, with_ibn=True, with_relu=True)  # 64, 64
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, kernel_size=3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            # Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),  # 99,64  2*hr+32+3
            Conv2dIBNormRelu(2 * hr_channels + 32 + 3, 2 * hr_channels, 3, stride=1, padding=1),  # MODIFY
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),  # 64,64
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),  # 64,32
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),  # 64,32
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),  # 32,32
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),  # 32,32
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),  # 96,32
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0),
            # 32,1
        )

    def forward(self, img, enc2x, enc4x, lr8x, inference=False):
        # 两次降采样
        img2x = F.interpolate(img, scale_factor=1 / 2, mode='bilinear', align_corners=False,
                              recompute_scale_factor=True)  # [1,3,512,512]--->[1,3,256,256]
        img4x = F.interpolate(img, scale_factor=1 / 4, mode='bilinear', align_corners=False,
                              recompute_scale_factor=True)  # [1,3,128,128]

        enc2x = self.tohr_enc2x(enc2x)  # [1,16,256,256]--->[1,32,256,256]  24
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x),  # 16+3
                                         dim=1))  # [1,3,256,256] + [1,32,256,256] ---> [1,35,256,256] ---> [1,32,128,128]   27-->24

        enc4x = self.tohr_enc4x(enc4x)  # [1,24,128,128] ---> [1,32,128,128]  24
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x),
                                         dim=1))  # [1,32,128,128] + [1,32,128,128] ---> [1,64,128,128] ---> [1,64,128,128]  48

        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear',
                             align_corners=False, recompute_scale_factor=True)  # [1,32,64,64] ---> [1,32,128,128]
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x),
                                        dim=1))  # [1,64,128,128]+[1,32,128,128]+[1,3,128,128] ---> [1,99,128,128] --->[1,32,128,128]   48+32+3==83

        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False,
                             recompute_scale_factor=True)  # [1,32,128,128]--->[1,32,256,256]
        hr2x = self.conv_hr2x(
            torch.cat((hr2x, enc2x), dim=1))  # [1,32,256,256] + [1,32,256,256] ---> [1,64,256,256]--->[1,32,256,256]

        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, f_channel):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(f_channel[0], hr_channels, 5, stride=1, padding=2)

        self.conv_f2x = hrRepConv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            hrRepConv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)

        return pred_matte


# ------------------------------------------------------------------------------
#  MODNet
# ------------------------------------------------------------------------------

class MODNet_auto(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, cfg=None, expansion=None, enc_channels=None, lr_channel=None, hr_channel=None,
                 f_channel=None, in_channels=3,
                 hr_channels=32,
                 backbone_arch='mobilenetv2_auto',
                 backbone_pretrained=True, deploy=False):
        super(MODNet_auto, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained
        self.deploy = deploy

        if self.deploy:
            self._switch_to_deploy()

        if cfg is None:
            cfg = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]
        if expansion is None:
            expansion = [None, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, None]
        else:
            expansion = [None] + expansion + [None]

        # if enc_channels is None:
        #     enc_channels = [16, 24, 32, 96, 1280]

        if lr_channel is None:
            lr_channel = [32, 96]
        if hr_channel is None:
            hr_channel = [16, 24]
        if f_channel is None:
            f_channel = [32]

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels, cfg, expansion)

        self.lr_branch = LRBranch(self.backbone, lr_channel)
        self.hr_branch = HRBranch(self.hr_channels, hr_channel)
        self.f_branch = FusionBranch(self.hr_channels, f_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()

    def _switch_to_deploy(self):
        def _apply_reparam(module):
            if isinstance(module, Conv2dIBNormRelu):
                module.switch_to_deploy()

        self.apply(_apply_reparam)

    def forward(self, img, inference=False):

        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_matte

    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)