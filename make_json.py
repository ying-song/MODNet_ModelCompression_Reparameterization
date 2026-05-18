import os
import json
import torch
import argparse

from src.models.backbones.repmobilenetv2_auto import RepInvertedResidual, RepConvBNRelu
from nni.compression.pytorch.utils import count_flops_params
from pruner.block import InvertResBlock, CBR, CIBRelu, LBlock
from pruner.prune import compute_weights, get_nums_of_keep_channels
from src.models.modnet_auto2 import MODNet_auto, RepConv2dIBNormRelu, SEBlock


def get_model_block(model):
    """
    获取模型的层结构信息，包括参数、输入输出通道数等。
    """
    backbone_blocks = []
    lr_blocks = []
    hr_blocks = []
    f_blocks = []

    layer_count = 0
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, RepInvertedResidual):
            backbone_blocks.append(InvertResBlock(name, list(module.state_dict().values())))
        elif isinstance(module, RepConvBNRelu):
            backbone_blocks.append(CBR(name, list(module.state_dict().values())))
        elif isinstance(module, SEBlock):
            lr_blocks.append(LBlock(name, list(module.state_dict().values())))
        elif isinstance(module, RepConv2dIBNormRelu):
            if layer_count < 3:
                lr_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            elif 3 <= layer_count < 16:
                hr_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            else:
                f_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            layer_count += 1

    return backbone_blocks + lr_blocks + hr_blocks + f_blocks


def get_pruning_cfg(blocks, ratio, threshold):
    """
    根据网络层的通道权重生成剪枝配置，返回各层保留的输出通道数。
    """
    model_out_cfg = []
    for block in blocks:
        if isinstance(block, CBR):
            model_out_cfg.append(compute_weights(block.weight, threshold))
        elif isinstance(block, InvertResBlock):
            nums_keep = max(compute_weights(block.output1_weight, threshold),
                            compute_weights(block.output2_weight, threshold))
            model_out_cfg.extend([nums_keep] * 2)
            model_out_cfg.append(compute_weights(block.output3_weight, threshold))
        elif isinstance(block, LBlock):
            model_out_cfg.append(int(model_out_cfg[-1] / 4))
        elif isinstance(block, CIBRelu):
            model_out_cfg.append(get_nums_of_keep_channels(ratio, block.output_channel))
    return model_out_cfg


def save_json(data, output_path):
    """
    保存数据为JSON文件。
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON file saved to {output_path}")


def main(pretrained_path, output_path, ratio=0.5, threshold=0.5):
    """
    主函数：加载网络和权重，生成JSON文件。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODNet_auto().to(device)

    # 加载预训练权重
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        raise FileNotFoundError(f"Pretrained weights not found at {pretrained_path}")

    # 获取模型层结构信息
    model_blocks = get_model_block(model)

    # 生成剪枝配置
    pruning_cfg = get_pruning_cfg(model_blocks, ratio, threshold)

    # 保存为JSON
    output_data = {
        "pruning_cfg": pruning_cfg,
        "model_blocks": [block.to_dict() for block in model_blocks]  # 假设block对象实现了to_dict方法
    }
    save_json(output_data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON file for MODNet pruning configuration")
    parser.add_argument("--pretrained_path", type=str, required=False, default="./pretrained/try_rep_m_mppm_r_0.5_t_0.4_0.3_longer_epoch61.pth",
                        help="Path to pretrained model weights")
    parser.add_argument("--output_path", type=str, required=False, default="./result/prune/modify_mppm_prune_twice/r_0.5_t0.4/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.3.json",
                        help="Path to save the JSON file")
    parser.add_argument("--ratio", type=float, default=0.5, help="Pruning ratio for HR and F branches")
    parser.add_argument("--threshold", type=float, default=0.3, help="Pruning threshold for backbone layers")

    args = parser.parse_args()
    main(args.pretrained_path, args.output_path, args.ratio, args.threshold)
