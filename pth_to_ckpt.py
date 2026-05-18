import torch

# 加载 .pth 文件中的 state_dict
state_dict = torch.load('model_save/modify1layer_rep_m_p_n_t_0027_lr0.0001_r_0.5_t_0.5/new_trimap_0014_lr0.001.pth')

# 将这些信息组合到一个字典中
checkpoint = {
    'state_dict': state_dict
}

# 保存为 .ckpt 文件
torch.save(checkpoint, 'pth_to_ckpt/modify1layer_rep_m_p_n_t_0027_lr0.0001_r_0.5_t_0.5/new_trimap_0014_lr0.001.ckpt')
