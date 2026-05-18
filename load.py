import torch
from src.models.modnet_auto import MODNet_auto
import sys, traceback

def log(exc_type, exc_value, exc_traceback):
    error_log_path = './error_log.txt'
    with open(error_log_path, "a")as error_file:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=error_file)
        error_file.write("\n" + "-" *80 + "\n")

sys.excepthook = log

model = MODNet_auto(backbone_pretrained=False)
checkpoint = torch.load("./model_save/modified_mppm/new_trimap_0017_lr0.01.pth")
new_state_dict = {}
for k,v in checkpoint.items():
    if k.startswith("module."):
        new_key = k[7:]
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
torch.save(new_state_dict,"./model_save/1.pth")


# import torch
# import re
# import argparse
#
#
# def extract_unexpected_keys(error_log_path):
#     """从错误日志中精确提取unexpected keys"""
#     unexpected_keys = []
#     with open(error_log_path, 'r') as f:
#         log_content = f.read()
#
#         # 使用更精确的正则表达式匹配unexpected keys部分
#         unexpected_section = re.search(
#             r"Unexpected key\(s\) in state_dict:\s*((?:[\"'][^\"']+[\"'],?\s*)+)",
#             log_content
#         )
#
#         if unexpected_section:
#             # 提取包含所有键的字符串
#             keys_text = unexpected_section.group(1).strip()
#
#             # 使用正则表达式精确提取每个键
#             keys_list = re.findall(r'[\'"]([^\'"]+)[\'"]', keys_text)
#
#             # 清理每个键：移除可能的逗号和空格
#             unexpected_keys = [key.strip().rstrip(',').strip() for key in keys_list]
#
#     return set(unexpected_keys)  # 返回集合确保唯一性
#
#
# def fix_state_dict(input_path, error_log_path, output_path):
#     """精确修复状态字典，只修改error_log中明确提到的unexpected keys"""
#     # 加载原始状态字典
#     state_dict = torch.load(input_path, map_location=torch.device('cpu'))
#     original_keys = set(state_dict.keys())
#
#     print(f"原始状态字典包含 {len(original_keys)} 个键")
#
#     # 提取需要修改的键
#     keys_to_fix = extract_unexpected_keys(error_log_path)
#     print(f"日志中列出 {len(keys_to_fix)} 个需要修复的键")
#
#     # 验证日志中的键是否确实存在于状态字典中
#     missing_in_state_dict = keys_to_fix - original_keys
#     if missing_in_state_dict:
#         print(f"警告: {len(missing_in_state_dict)} 个日志中的键在状态字典中不存在")
#         print("前5个缺失键:", list(missing_in_state_dict)[:5])
#
#     # 创建修复后的状态字典
#     fixed_state_dict = {}
#     keys_modified = 0
#     keys_skipped = 0
#
#     for key in state_dict.keys():
#         # 仅修改明确列在日志中的键
#         if key in keys_to_fix:
#             # 移除"module."前缀
#             new_key = key.replace("module.", "", 1)
#             fixed_state_dict[new_key] = state_dict[key]
#             keys_modified += 1
#             if keys_modified <= 5:  # 只打印前5个修改示例
#                 print(f"修改键: {key} -> {new_key}")
#         else:
#             # 保持原键
#             fixed_state_dict[key] = state_dict[key]
#             keys_skipped += 1
#
#     # 保存修复后的状态字典
#     torch.save(fixed_state_dict, output_path)
#     print(f"修复完成: 修改了 {keys_modified} 个键, 保留了 {keys_skipped} 个键")
#     print(f"修复后的状态字典已保存至: {output_path}")
#
#     # 验证修复后的键数量
#     fixed_keys = set(fixed_state_dict.keys())
#     keys_removed = original_keys - fixed_keys
#     keys_added = fixed_keys - original_keys
#
#     print("\n验证结果:")
#     print(f"原始键数: {len(original_keys)}")
#     print(f"修复后键数: {len(fixed_keys)}")
#     print(f"移除的键数: {len(keys_removed)}")
#     print(f"新增的键数: {len(keys_added)}")
#
#     if keys_removed:
#         print(f"注意: {len(keys_removed)} 个键被移除")
#     if keys_added:
#         print(f"注意: {len(keys_added)} 个新键被添加")
#
#     # 返回修改的键数量用于验证
#     return keys_modified
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='精确修复状态字典键名')
#     parser.add_argument('--input', type=str, required=False,
#                         default="./model_save/pruned_once_modified_rep_all_layers_all_fixed_bn/new_trimap_0073_lr0.0001.pth",
#                         help='原始状态字典文件路径')
#     parser.add_argument('--error_log', type=str, required=False, default="./error_log.txt",
#                         help='包含错误信息的日志文件路径')
#     parser.add_argument('--output', type=str, required=False, default="./model_save/1.pth",
#                         help='修复后状态字典保存路径')
#
#     args = parser.parse_args()
#
#     modified_keys = fix_state_dict(args.input, args.error_log, args.output)
#
#     # 如果没有修改任何键，显示警告
#     if modified_keys == 0:
#         print("\n警告: 没有修改任何键! 请检查错误日志格式和内容")
#         print("建议: 确认错误日志中包含 'Unexpected key(s) in state_dict' 部分")
