import torch
from thop import profile
from src.models.modnet_auto2 import MODNet_auto  # 确保导入模型类
import json

# 从剪枝时生成的 JSON 加载配置
with open('./result/prune/modify_mppm_prune_once/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.4.json') as f:
    prune_info = json.load(f)

# 用配置重建网络
model = MODNet_auto(
    cfg=prune_info['new_cfg'],
    expansion=prune_info['new_expansion_cfg'],
    lr_channel=prune_info['new_lr_channels'],
    hr_channel=prune_info['new_hr_channels'],
    f_channel=prune_info['new_f_channels'],
    hr_channels=int(32 * (1 - prune_info['ratio'])),  # original_channels_of_hr_branch=32
    backbone_pretrained=False
)

# 2. 加载状态字典
state_dict = torch.load('./model_save/1.pth')
model.load_state_dict(state_dict)

# 3. 设置为评估模式
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)

# 测试剪枝模型
with torch.no_grad():
    flops_pruned, params_pruned = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )

print(f"参数量：{params_pruned / 1e6:.2f}M")
print(f"FLOPs：{flops_pruned / 1e6:.2f}M")

