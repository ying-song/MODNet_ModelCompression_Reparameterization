# uploads/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np
import os

class MODNet_auto(nn.Module):
    def __init__(self, cfg, expansion, lr_channel, hr_channel, f_channel, hr_channels, backbone_pretrained=False):
        super(MODNet_auto, self).__init__()
        # 模型定义代码
        pass

def predit_matte(modnet: nn.Module, im: Image):
    # 定义图像到张量的转换
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # 定义超参数
    ref_size = 512
    modnet.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        # 统一图像通道数为3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        im = Image.fromarray(im)
        # 将图像转换为 PyTorch 张量
        im = im_transform(im)

        # 添加 mini-batch 维度
        im = im[None, :, :, :]

        # 调整图像大小以适应输入
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        im = im.to(device)

        # 推理
        _, _, matte = modnet(im)
        # 调整和保存 matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        return matte

def pred_result(img, matte):
    prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')

    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    prd_img = cv.cvtColor(np.asarray(prd_img), cv.COLOR_RGB2BGR)
    res = cv.add(img, 255 - prd_img)
    h, w = img.shape[:2]
    res = cv.resize(res, (int(w / 4), int(h / 4)))
    cv.imshow('img', res)
    cv.waitKey(0)
    cv.destroyAllWindows()

def infer_images(model, file_dir, save_dir):
    files = list(os.walk(file_dir))[0][2:][0]

    for file_name in files:
        img = Image.open(file_dir + file_name)

        matte = predit_matte(model, img)

        prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        prd_img = np.asarray(prd_img)

        img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
        mask = cv.merge([prd_img, prd_img, prd_img])
        res = cv.add(img, 255 - mask)
        h, w = img.shape[:2]
        cv.imwrite(save_dir + "/" + file_name, res)

def images_with_single_model(model, model_path, file_dir, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    weights = torch.load(model_path, map_location=torch.device('cuda:0'))
    model.load_state_dict(weights)

    infer_images(model, file_dir, save_path)

def infer_images_with_models(model, model_dir_path, file_dir, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filenames = os.listdir(model_dir_path)
    for filename in filenames:
        ckpt_path = model_dir_path + filename
        weights = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
        model.load_state_dict(weights)

        save_dir = save_path + ckpt_path.split('/')[-1][:-4]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        infer_images(model, file_dir, save_dir)
