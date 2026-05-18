import torch
import torch.nn as nn
from torch.nn import init
from src.models.modnet_auto2_rep5x5 import MODNet_auto

def fix_state_dict(state_dict):
    """去除key中的'module.'前缀并过滤不需要的key"""
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除'module.'前缀
        if k.startswith('module.'):
            k = k[7:]

        if not any(unexpected_key in k for unexpected_key in [
            "lr_branch.conv_lr.scale_dense",
            "lr_branch.conv_lr.scale_1x1",
            "lr_branch.conv_lr.rbr_dense.weight",
            "lr_branch.conv_lr.rbr_1x1.weigh"
        ]):
            new_state_dict[k] = v
    return new_state_dict


def initialize_missing_keys(model, pretrained_dict):
    """初始化模型中缺失的参数"""
    model_dict = model.state_dict()

    # 识别缺失的key
    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())

    # 对缺失的参数进行初始化
    for key in missing_keys:
        param = model_dict[key]

        # 根据参数类型选择初始化方式
        if 'weight' in key:
            if len(param.shape) > 1:  # 非bias的权重
                if 'conv' in key or 'fc' in key:
                    # 对卷积层和全连接层使用kaiming初始化方法
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bnorm' in key:
                    init.normal_(param, mean=1.0, std=0.02)
            else:
                init.constant_(param, 1.0)
        elif 'bias' in key:
            init.constant_(param, 0.0)
        elif 'running_mean' in key:
            init.constant_(param, 0.0)
        elif 'running_var' in key:
            init.constant_(param, 1.0)

        print(f'Initialized missing key: {key}')

    return missing_keys


def process_checkpoint(input_path, output_path, model):
    # 加载原始checkpoint
    original_sd = torch.load(input_path)

    # 处理state_dict
    processed_sd = fix_state_dict(original_sd)

    # 初始化缺失参数并过滤不需要的key
    model_sd = model.state_dict()

    # 1. 过滤掉不需要的key（pretrained中存在但model不需要的）
    filtered_sd = {k: v for k, v in processed_sd.items() if k in model_sd}

    # 2. 添加model需要但pretrained中没有的key（已初始化）
    missing_keys = initialize_missing_keys(model, filtered_sd)

    # 3. 最终合并（优先使用pretrained参数，缺失的用初始化后的参数）
    final_sd = {**model.state_dict(), **filtered_sd}
    final_sd.update(filtered_sd)

    model_keys = set(model.state_dict().keys())
    final_keys = set(final_sd.keys())

    if model_keys != final_keys:
        print("misatch")
        print("extra keys:",final_keys - model_keys)
        print("missing keys:", model_keys - final_keys)

    # 保存处理后的checkpoint
    torch.save(final_sd, output_path)
    print(f'Successfully processed and saved to {output_path}')


# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    model = torch.nn.DataParallel(MODNet_auto(backbone_pretrained=False))

    # 输入输出路径（替换为实际路径）
    # input_checkpoint = "./model_save/modify_mppm_r_0.5_t_0.4_0.3_longer/new_trimap_0061_lr0.0001.pth"
    input_checkpoint = "./model_save/pruned_once_modified_yuanban/new_trimap_0095_lr1e-05.pth"
    output_checkpoint = "./model_save/pruned_once_rep_lr_all5x5layers.pth"

    # 执行处理
    process_checkpoint(input_checkpoint, output_checkpoint, model)
