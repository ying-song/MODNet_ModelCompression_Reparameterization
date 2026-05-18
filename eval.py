"""
# load official modnet
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)
ckpt_path = './pretrained/modify_modnet_photographic_portrait_matting'

# load our own trained modnet
# modnet = MODNet_auto(backbone_pretrained=False)
# modnet = nn.DataParallel(modnet)
# ckp_pth = '../pretrained/our_modnet.ckpt'
"""
import argparse

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from glob import glob
import json
import sys
sys.path.append('/home/xxfs/MODNet-ModelCompression')

from src.models.modnet import MODNet
from src.models.modnet_auto import MODNet_auto

from infer import predit_matte


def cal_mad(pred, gt):
    diff = pred - gt
    diff = np.abs(diff)
    mad = np.mean(diff)
    return mad


def cal_mse(pred, gt):
    diff = pred - gt
    diff = diff ** 2
    mse = np.mean(diff)
    return mse


def load_eval_dataset(dataset_root_dir='./src/datasets/PPM-100'):
    image_path = dataset_root_dir + '/val/fg/*'
    matte_path = dataset_root_dir + '/val/alpha/*'
    image_file_name_list = glob(image_path)
    image_file_name_list = sorted(image_file_name_list)
    matte_file_name_list = glob(matte_path)
    matte_file_name_list = sorted(matte_file_name_list)

    return image_file_name_list, matte_file_name_list


def eval(modnet: MODNet, dataset):
    mse = total_mse = 0.0
    mad = total_mad = 0.0
    cnt = 0

    for im_pth, mt_pth in zip(dataset[0], dataset[1]):
        im = Image.open(im_pth)
        pd_matte = predit_matte(modnet, im)

        gt_matte = Image.open(mt_pth)
        gt_matte = np.asarray(gt_matte) / 255

        total_mse += cal_mse(pd_matte, gt_matte)
        total_mad += cal_mad(pd_matte, gt_matte)

        cnt += 1
    if cnt > 0:
        mse = total_mse / cnt
        mad = total_mad / cnt

    return mse, mad


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=False, default='./model_save/rep_hr_layer_modify_mppm_r_0.5_t_0.4_0.3_longer/new_trimap_0001_lr0.01.pth',
                        help='path of the checkpoint that will be evaluated')
    parser.add_argument('--prune-info', type=str, required=False, default='./result/prune/modify_mppm_prune_twice/r_0.5_t0.4/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.3.json',
                        help='path of the prune info')
    args = parser.parse_args()

    # check input argumentsd
    ckpt_path = args.ckpt_path
    prune_info = args.prune_info

    # load ppm-100 dataset
    dataset_dir = load_eval_dataset('./src/datasets/PPM-100')

    # load pruned model
    prune_info = json.load(open(prune_info))
    ratio = prune_info['ratio']
    threshold = prune_info['threshold']
    my_cfg = prune_info['new_cfg']
    my_expansion_cfg = prune_info['new_expansion_cfg']
    my_hr_channels = prune_info['new_hr_channels']
    my_lr_channels = prune_info['new_lr_channels']
    my_f_channels = prune_info['new_f_channels']

    modnet = MODNet_auto(cfg=my_cfg, expansion=my_expansion_cfg, lr_channel=my_lr_channels,
                         hr_channel=my_hr_channels,
                         f_channel=my_f_channels,
                         hr_channels=int(32 * (1 - ratio)),
                         backbone_pretrained=False, deploy=True)

    modnet = nn.DataParallel(modnet)
    # func1 1, eval images with single model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modnet = modnet.to(device)
    weights = torch.load(ckpt_path, map_location=device)
    modnet.load_state_dict(weights, strict=True)
    modnet.eval()  # 切换到推理模式

    # weights = torch.load(ckpt_path, map_location=torch.device('cuda'))
    # modnet.load_state_dict(weights)
    print("Starting......")
    mse, mad = eval(modnet, dataset_dir)
    print(f' mse: {mse:6f}, mad: {mad:6f}')

    # func2, eval images with multiple models in dir
    # # to find the best model with the lowest mse&mad
    # ckp_dir = ''
    # ckp_pth_set = os.listdir(ckp_dir)
    # ckp_pth_set.sort(key=lambda x: x[-6:-4])
    # for ckp_name in ckp_pth_set:
    #     ckp_pth = ckp_dir + ckp_name
    #     weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    #     modnet.load_state_dict(weights)
    #     mse, mad = eval(modnet, dataset_dir)
    #     print(f'{ckp_name}    mse: {mse:6f}, mad: {mad:6f}')
