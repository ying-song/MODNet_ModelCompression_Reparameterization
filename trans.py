import torch
from src.models.modnet_auto import MODNet_auto

# 加载模型和权重
model = torch.nn.DataParallel(MODNet_auto(backbone_pretrained=False))
torch_weight = model.state_dict()

# 加载预训练的模型权重
weights = torch.load("./pretrained/modnet_photographic_portrait_matting.ckpt")

torch_weight_keys = list(torch_weight.keys())
weights_values = list(weights.values())
new_weights = {}
for i in range(len(torch_weight_keys)):
    new_weights[torch_weight_keys[i]] = weights_values[i]

torch_weight.update(new_weights)

# 保存更新后的模型
model.load_state_dict(torch_weight)
torch.save(model.state_dict(), "./pretrained/modify_mppm_modnet.ckpt")
