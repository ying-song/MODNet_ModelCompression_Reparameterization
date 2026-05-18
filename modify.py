import torch
import re
from src.models.modnet_auto2_rep5x5 import MODNet_auto

def parse_mismatch_file(mismatch_file):
    """
    解析包含 size mismatch 错误的文本文件。

    :param mismatch_file: str, 包含 size mismatch 报错的文件路径
    :return: dict, 错误信息的映射，键为参数名，值为 (checkpoint_shape, model_shape)
    """
    mismatch_info = {}
    with open(mismatch_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # match = re.match(r"\s*size mismatch for (.+?):.*torch\.Size\((.+?)\).*torch\.Size\((.+?)\)",line)
            match = re.match(r"\s*size mismatch for (.+?):.*torch\.Size\(\[([0-9, ]+)\]\).*torch\.Size\(\[([0-9, ]+)\]\)", line)
            if match:
                param_name = match.group(1)
                ckpt_shape = tuple(map(int, re.findall(r"\d+", match.group(2))))
                model_shape = tuple(map(int, re.findall(r"\d+", match.group(3))))
                mismatch_info[param_name] = (ckpt_shape, model_shape)
                print(f"match found:{param_name}, {ckpt_shape}, {model_shape}")
            else:
                print(f"no match for line:{line}")
    return mismatch_info

def adjust_checkpoint_weights_from_mismatch(checkpoint_path, model, mismatch_file, save_path):
    """
    根据 mismatch 文件调整 checkpoint 中的权重。

    :param checkpoint_path: str, 原始 checkpoint 路径
    :param model: torch.nn.Module, 当前模型
    :param mismatch_file: str, 包含 size mismatch 报错的文件路径
    :param save_path: str, 调整后的 checkpoint 保存路径
    """
    # 加载 checkpoint 和模型参数
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get('state_dict', checkpoint)  # 支持直接或嵌套的 state_dict
    model_dict = model.state_dict()

    # 解析 mismatch 文件
    mismatch_info = parse_mismatch_file(mismatch_file)
    print(mismatch_info)

    for key, (ckpt_shape, model_shape) in mismatch_info.items():
        if key in state_dict:
            print(f"Adjusting {key}: checkpoint shape {ckpt_shape} -> model shape {model_shape}")

            # 获取权重并调整形状
            weight = state_dict[key]
            for dim in range(len(ckpt_shape)):
                if ckpt_shape[dim] > model_shape[dim]:  # 裁剪多余维度
                    weight = weight.narrow(dim, 0, model_shape[dim])
                    print(f"croped weight{key}: {weight.shape}")
                elif ckpt_shape[dim] < model_shape[dim]:  # 不支持扩展维度
                    raise ValueError(f"Model has more dimensions ({model_shape}) than checkpoint ({ckpt_shape}).")

            state_dict[key] = weight

    # 保存调整后的权重
    torch.save(state_dict, save_path)
    print(f"Modified checkpoint saved to {save_path}")

model = MODNet_auto(backbone_pretrained=False)  # 替换为您的模型实例
adjust_checkpoint_weights_from_mismatch(
    checkpoint_path="./model_save/pruned_once_rep_lr_all5x5layers.pth",
    model=model,
    mismatch_file="./error_log.txt",  # 替换为包含报错内容的 txt 文件路径
    save_path="./model_save/pruned_once_rep_lr_all5x5layers.pth"
)

