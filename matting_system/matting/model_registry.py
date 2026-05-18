import os
import json
import torch


class ModelConfig:
    """模型配置管理类"""

    # 模型类型定义
    DEFAULT = 'default'
    HIGH_QUALITY = 'high_quality'
    LIGHTWEIGHT = 'lightweight'

    # 各模型的配置信息
    MODELS = {
        DEFAULT: {
            'name': '默认模型',
            'description': '平衡性能与速度的默认模型',
            'prune_info_path': r'D:\MODNet-ModelCompression\result\prune\modify_mppm_prune_once\modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.4.json',
            'ckp_pth': r'D:\MODNet-ModelCompression\model_save\pruned_once_modified_rep_flayers\new_trimap_0030_lr0.001.pth',
            'model_file': 'modnet_auto2',
            'deploy': True,
        },
        HIGH_QUALITY: {
            'name': '高质量模型',
            'description': '精度最高但速度较慢的高质量模型',
            'prune_info_path': r'D:\MODNet-ModelCompression\result\prune\modify_modnet_photographic_portrait_matting\modnet_p_new_trimap_0027_lr0.0001_ratio_0_thresh_1e-05.json',
            'ckp_pth': r'D:\MODNet-ModelCompression\model_save\modified_mppm\new_trimap_0017_lr0.01.pth',
            'model_file': 'modnet_auto',
            'deploy': False,
        },
        LIGHTWEIGHT: {
            'name': '轻量化模型',
            'description': '经过剪枝的轻量化模型，速度快',
            'prune_info_path': r'D:\MODNet-ModelCompression\result\prune\modify_mppm_prune_once\modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.4.json',
            'ckp_pth': r'D:\MODNet-ModelCompression\model_save\pruned_once_modified_yuanban\new_trimap_0095_lr1e-05.pth',
            'model_file': 'modnet_auto',
            'deploy': False,
        },
    }

    @classmethod
    def get_model_config(cls, model_type):
        """获取指定模型的配置信息"""
        if model_type not in cls.MODELS:
            raise ValueError(f"未知的模型类型：{model_type}")
        return cls.MODELS[model_type]

    @classmethod
    def get_all_models(cls):
        """获取所有可用模型的信息"""
        return [
            {
                'type': key,
                'name': info['name'],
                'description': info['description']
            }
            for key, info in cls.MODELS.items()
        ]

    @classmethod
    def load_model(cls, model_type):
        """加载指定类型的模型"""
        import importlib

        config = cls.get_model_config(model_type)

        # 动态导入对应的模型文件
        model_module = importlib.import_module(f'src.models.{config["model_file"]}')
        MODNet_auto = model_module.MODNet_auto

        # 加载剪枝信息
        with open(config['prune_info_path'], 'r') as f:
            prune_info = json.load(f)

        ratio = prune_info['ratio']
        my_cfg = prune_info['new_cfg']
        my_expansion_cfg = prune_info['new_expansion_cfg']
        my_hr_channels = prune_info['new_hr_channels']
        my_lr_channels = prune_info['new_lr_channels']
        my_f_channels = prune_info['new_f_channels']

        # 根据模型文件决定是否传递 deploy 参数
        # modnet_auto2.py 支持 deploy 参数，modnet_auto.py 不支持
        if config['model_file'] == 'modnet_auto2':
            # 支持 deploy 参数的版本
            model = MODNet_auto(
                cfg=my_cfg,
                expansion=my_expansion_cfg,
                lr_channel=my_lr_channels,
                hr_channel=my_hr_channels,
                f_channel=my_f_channels,
                hr_channels=int(32 * (1 - ratio)),
                backbone_pretrained=False,
                deploy=config['deploy']
            )
        else:
            # 不支持 deploy 参数的版本
            model = MODNet_auto(
                cfg=my_cfg,
                expansion=my_expansion_cfg,
                lr_channel=my_lr_channels,
                hr_channel=my_hr_channels,
                f_channel=my_f_channels,
                hr_channels=int(32 * (1 - ratio)),
                backbone_pretrained=False
            )

        # 如果不是默认模型，需要包装为 DataParallel
        if model_type != cls.DEFAULT:
            model = torch.nn.DataParallel(model)

        # 加载权重
        if torch.cuda.is_available():
            model = model.cuda()
            weights = torch.load(config['ckp_pth'])
        else:
            weights = torch.load(config['ckp_pth'], map_location=torch.device('cpu'))

        if not isinstance(model, torch.nn.DataParallel):
            # 当前模型不是 DataParallel，但权重是
            if any(key.startswith('module.') for key in weights.keys()):
                # 移除 module.前缀
                weights = {k.replace('module.', ''): v for k, v in weights.items()}

        model.load_state_dict(weights, strict=True)
        model.eval()

        return model, config