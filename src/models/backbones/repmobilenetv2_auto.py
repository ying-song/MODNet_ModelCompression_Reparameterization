from math import sqrt
import torch, gc
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvBNRelu(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding):
        super(ConvBNRelu, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, stride, padding, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

    def forward(self, x):
        return self.cbr(x)

# 初版修改1.0
# class RepConvBNRelu(nn.Module):
#     def __init__(self, inp, oup, kernel, stride, padding):
#         super(RepConvBNRelu, self).__init__()
#         self.kernel = kernel
#         self.deploy = False
#
#         # 仅当kernel为3x3时进行重参数化
#         if kernel == 3:
#             # 重参数化分支
#             self.rbr_dense = nn.Conv2d(inp, oup, kernel, stride, padding, bias=False)
#             self.rbr_1x1 = nn.Conv2d(inp, oup, 1, stride, 0, bias=False)
#
#             # 各分支的缩放参数
#             self.scale_dense = nn.Parameter(torch.ones(oup, 1, 1))
#             self.scale_1x1 = nn.Parameter(torch.ones(oup, 1, 1))
#
#             # 恒等分支（仅当输入输出通道数相同且stride为1时）
#             if inp == oup and stride == 1:
#                 self.rbr_identity = nn.Identity()
#                 self.scale_identity = nn.Parameter(torch.ones(oup, 1, 1))
#             else:
#                 self.rbr_identity = None
#
#             # 批归一化和ReLU
#             self.bn = nn.BatchNorm2d(oup)
#             self.relu = nn.ReLU6(inplace=True)
#         else:
#             # 对于非3x3卷积核，使用标准卷积
#             self.standard_conv = nn.Sequential(
#                 nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU6(inplace=True)
#             )
#
#     def forward(self, x):
#         # 对于非3x3卷积核，直接使用标准卷积
#         if self.kernel != 3:
#             return self.standard_conv(x)
#
#         # 训练模式下的多分支处理
#         if not self.deploy:
#             dense_out = self.rbr_dense(x) * self.scale_dense
#             x1x1 = self.rbr_1x1(x) * self.scale_1x1
#
#             # 填充以对齐形状
#             if x1x1.shape != dense_out.shape:
#                 diff_h = dense_out.shape[2] - x1x1.shape[2]
#                 diff_w = dense_out.shape[3] - x1x1.shape[3]
#                 x1x1 = F.pad(x1x1, [diff_w // 2, diff_w - diff_w // 2,
#                                     diff_h // 2, diff_h - diff_h // 2])
#
#             # 恒等分支
#             if self.rbr_identity is not None:
#                 id_out = x * self.scale_identity
#                 if id_out.shape != dense_out.shape:
#                     diff_h = dense_out.shape[2] - id_out.shape[2]
#                     diff_w = dense_out.shape[3] - id_out.shape[3]
#                     id_out = F.pad(id_out, [diff_w // 2, diff_w - diff_w // 2,
#                                             diff_h // 2, diff_h - diff_h // 2])
#             else:
#                 id_out = 0
#
#             # 分支求和
#             out = dense_out + x1x1 + id_out
#             out = self.bn(out)
#             return self.relu(out)
#
#         # 部署模式下的融合卷积
#         return self.relu(self.rbr_reparam(x))
#
#     def get_equivalent_kernel_bias(self):
#         # 仅当kernel为3x3时进行融合
#         if self.kernel != 3:
#             return None, None
#
#         # 融合卷积核和偏置
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense, self.scale_dense)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1, self.scale_1x1)
#
#         # 恒等分支卷积核
#         kernelid, biasid = (self._fuse_bn_tensor(None, self.scale_identity)
#                             if self.rbr_identity is not None else (0, 0))
#
#         return (kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
#                 bias3x3 + bias1x1 + biasid)
#
#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         return F.pad(kernel1x1, [1, 1, 1, 1])
#
#     def _fuse_bn_tensor(self, conv, scale):
#         if conv is None:
#             # 恒等分支处理
#             input_dim = self.rbr_dense.in_channels // self.rbr_dense.groups
#             kernel_value = np.zeros((self.rbr_dense.in_channels, input_dim, 3, 3), dtype=np.float32)
#             for i in range(self.rbr_dense.in_channels):
#                 kernel_value[i, i % input_dim, 1, 1] = 1
#             kernel = torch.from_numpy(kernel_value).to(scale.device)
#             bias = 0
#         else:
#             kernel = conv.weight * scale.reshape(-1, 1, 1, 1)
#             bias = 0
#
#         return kernel, bias
#
#     def switch_to_deploy(self):
#         # 仅对3x3卷积核进行转换
#         if self.kernel == 3 and not self.deploy:
#             # 融合卷积核
#             kernel, bias = self.get_equivalent_kernel_bias()
#
#             # 创建带融合权重的新卷积
#             self.rbr_reparam = nn.Conv2d(
#                 self.rbr_dense.in_channels,
#                 self.rbr_dense.out_channels,
#                 kernel_size=3,
#                 stride=self.rbr_dense.stride,
#                 padding=1,
#                 bias=False
#             )
#             self.rbr_reparam.weight.data = kernel
#
#             # 移除训练分支
#             self.rbr_dense = None
#             self.rbr_1x1 = None
#             self.rbr_identity = None
#             self.bn = None
#
#             self.deploy = True

# 修改2.0

class RepConvBNRelu(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, padding=1):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.kernel_size = kernel_size
        self.stride = stride
        self.deploy = False

        # 仅对3x3卷积进行重参数化
        if kernel_size == 3:
            # 训练用多分支结构
            self.rbr_3x3 = nn.Conv2d(inp, oup, 3, stride, padding, bias=False)
            self.rbr_1x1 = nn.Conv2d(inp, oup, 1, stride, 0, bias=False)
            self.rbr_identity = nn.Identity() if (inp == oup and stride == 1) else None

            # 可学习缩放系数
            self.scale_3x3 = nn.Parameter(torch.ones(oup, 1, 1))
            self.scale_1x1 = nn.Parameter(torch.ones(oup, 1, 1))
            self.scale_identity = nn.Parameter(torch.ones(oup, 1, 1)) if self.rbr_identity else None

            # 原BN层
            self.bn = nn.BatchNorm2d(oup)
            self.act = nn.ReLU6(inplace=True)
        else:
            # 非3x3保持原结构
            self.cbr = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

    def fuse(self):
        if self.kernel_size != 3: return

        # 合并所有分支的权重
        weight_3x3 = self.rbr_3x3.weight * self.scale_3x3.view(-1,1,1,1)
        weight_1x1 = F.pad(self.rbr_1x1.weight, [1,1,1,1]) * self.scale_1x1.view(-1,1,1,1)
        total_weight = weight_3x3 + weight_1x1

        #合并Identity分支（仅条件满足时）
        if self.rbr_identity and self.stride == 1:
            identity_weight = torch.eye(self.inp).view(self.inp, self.inp, 1, 1)
            identity_weight = F.pad(identity_weight, [1,1,1,1])
            total_weight += identity_weight * self.scale_identity.view(-1,1,1,1)

        # 融合BN到卷积中
        fused_weight, fused_bias = self._fuse_bn(self.rbr_3x3.weight, self.bn)
        # 创建部署用卷积
        self.rbr_reparam = nn.Conv2d(
            self.inp,
            self.oup,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True
        )
        self.rbr_reparam.weight.data = fused_weight
        self.rbr_reparam.bias.data = fused_bias
        self.deploy = True

    def _fuse_bn(self, conv_weight, bn):
        fused_weight = conv_weight * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)
        fused_bias = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
        return fused_weight, fused_bias

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        if self.kernel_size !=3:
            return self.cbr(x)

        out_3x3 = self.rbr_3x3(x) * self.scale_3x3
        out_1x1 = self.rbr_1x1(x) * self.scale_1x1

        id_out = 0
        if self.rbr_identity and self.stride == 1:
            id_out = x * self.scale_identity

        fused = self.bn(out_3x3 + out_1x1 + id_out)
        return self.act(fused)


# 二版修改，opera在线卷积
# class RepConvBNRelu(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
#         super(RepConvBNRelu, self).__init__()
#         self.deploy = False
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = max(1, groups)
#
#         # Convolutional layers without BatchNorm
#         self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=self.groups, bias=False)
#         self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=self.groups, bias=False)
#
#         # Scaling layers for each branch
#         self.scale_dense = nn.Parameter(torch.ones(out_channels, 1, 1))
#         self.scale_1x1 = nn.Parameter(torch.ones(out_channels, 1, 1))
#
#         # Identity branch setup
#         if in_channels == out_channels and stride == 1:
#             self.rbr_identity = True
#             self.scale_identity = nn.Parameter(torch.ones(out_channels, 1, 1))
#         else:
#             self.rbr_identity = None
#
#         # BN layer after all branches
#         self.bn_after_add = nn.BatchNorm2d(out_channels) if not self.deploy else None
#         self.relu = nn.ReLU6(inplace=True)
#
#     def forward(self, x):
#         # Deploy mode uses fused convolutions
#         if self.deploy:
#             return self.relu(self.rbr_reparam(x))
#
#         # Each branch with scaling applied
#         dense_out = self.rbr_dense(x) * self.scale_dense
#         x1x1 = self.rbr_1x1(x) * self.scale_1x1
#
#         # Padding for shape alignment
#         if x1x1.shape != dense_out.shape:
#             diff_h = dense_out.shape[2] - x1x1.shape[2]
#             diff_w = dense_out.shape[3] - x1x1.shape[3]
#             x1x1 = F.pad(x1x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
#
#         if self.rbr_identity is not None:
#             id_out = x * self.scale_identity
#             if id_out.shape != dense_out.shape:
#                 diff_h = dense_out.shape[2] - id_out.shape[2]
#                 diff_w = dense_out.shape[3] - id_out.shape[3]
#                 id_out = F.pad(id_out, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
#         else:
#             id_out = 0
#
#         # Sum of branches with additional BN
#         out = dense_out + x1x1 + id_out
#         if self.bn_after_add is not None:
#             out = self.bn_after_add(out)
#
#         return self.relu(out)
#
#     # Fusion method for deployment
#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense, self.scale_dense)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1, self.scale_1x1)
#         kernelid, biasid = (self._fuse_bn_tensor(None, self.scale_identity) if self.rbr_identity else (0, 0))
#
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
#
#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return F.pad(kernel1x1, [1, 1, 1, 1])
#
#     def _fuse_bn_tensor(self, conv, scale):
#         if conv is None:
#             input_dim = self.in_channels // self.groups
#             kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
#             for i in range(self.in_channels):
#                 kernel_value[i, i % input_dim, 1, 1] = 1
#             kernel = torch.from_numpy(kernel_value).to(scale.device)
#             bias = 0
#         else:
#             kernel = conv.weight * scale.reshape(-1, 1, 1, 1)
#             bias = 0
#
#         return kernel, bias

# 未修改，原版
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expansion, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class RepMobileNetV2Auto(nn.Module):
    def __init__(self, in_channels, cfg=None, expansion_cfg=None, num_classes=1000, deploy=False):
        super(RepMobileNetV2Auto, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if cfg is None:
            cfg = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]
        if expansion_cfg is None:
            expansion_cfg = [None, 1] + [6] * 16 + [None]

        interverted_residual_setting = [None,
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        None]

        for i, v in enumerate(cfg):
            if i == 0 or i == len(cfg) - 1:  # in & out
                interverted_residual_setting[i] = v
            else:
                interverted_residual_setting[i][1] = v

        for i, v in enumerate(expansion_cfg):
            if i == 0 or i == len(cfg) - 1:
                continue
            interverted_residual_setting[i][0] = v

        # 1. building first layer
        input_channel, last_channel = interverted_residual_setting[0], interverted_residual_setting[-1]
        self.last_channel = last_channel
        self.features = [RepConvBNRelu(self.in_channels, input_channel, 3, 2, 1)]

        # 2. building inverted residual blocks
        for t, c, n, s in interverted_residual_setting[1:-1]:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, expansion=t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expansion=t))
                input_channel = output_channel

        # 3.building last several layers
        self.features.append(ConvBNRelu(input_channel, self.last_channel, 1, 1, 0))

        # make it nn.Sequential
        # Connecting the three parts of input features, inverse residuals, and output features
        self.features = nn.Sequential(*self.features)

        if self.num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Stage1
        x = self.features[0](x)
        x = self.features[1](x)
        # Stage2
        x = self.features[2](x)
        x = self.features[3](x)
        # Stage3
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        # Stage4
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        # Stage5
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)

        # Classification
        if self.num_classes is not None:
            x = x.mean(dim=(2, 3))
            x = self.classifier(x)

        # Output
        return x

    def _load_pretrained_model(self, pretrained_file):
        pretrain_dict = torch.load(pretrained_file, map_location='cpu')
        model_dict = {}
        state_dict = self.state_dict()
        print("[RepMobileNetV2] Loading pretrained model...")
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                print(k, "is ignored")
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
