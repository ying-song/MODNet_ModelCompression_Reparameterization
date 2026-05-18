import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# from src.models.modnet import MODNet
from src.models.modnet_auto import MODNet_auto
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import os
import json
import time
def predit_matte(modnet: MODNet_auto, im: Image):
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # define hyper-parameters
    ref_size = 512

    modnet.eval()

    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    im = Image.fromarray(im)
    # convert image to PyTorch tensor
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
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

    # inference
    # _, _, matte = modnet(im.cuda().half() if torch.cuda.is_available() else im, True)
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)


    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte

def infer_one_image(model, img_path):
    start_time = time.time()

    img = Image.open(img_path)
    # img=img.resize((500,500))
    matte = predit_matte(model, img)
    # h, w = matte.shape[:2]
    # res = cv.resize(matte, (int(w / 5), int(h / 5)))
    # 保持与原始图片相同的尺寸
    h, w = img.size

    prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')

    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    prd_img = cv.cvtColor(np.asarray(prd_img), cv.COLOR_RGB2BGR)
    res = cv.add(img, 255 - prd_img)
    h, w = img.shape[:2]
    # res = cv.resize(res, (int(w / 4), int(h / 4)))
    # path = "./the_infer_img_with_prue/" + img_path.split('/')[-1]
    # 推理结果保存路径
    path1 = "D:\\MODNet-ModelCompression\\infer_result\\a_modified_mppm"
    # linux上用下三行代码
    # if not os.path.exists(path1):
    #     os.makedirs(path1)
    # path = path1 + img_path.split('/')[-1]

    # Windows上用以下代码
    abs_path1 = os.path.abspath(path1)  # 获取绝对路径用于调试
    if not os.path.exists(path1):
        os.makedirs(path1)

    # 使用 os.path.basename 获取文件名，更可靠地处理路径
    filename = os.path.basename(img_path)
    path = os.path.join(path1, filename)  # 使用 os.path.join 构建路径
    # 至此

    elapsed = time.time() - start_time  # 计算处理时间

    cv.imwrite(path, res)

    return elapsed  # 返回处理时间

    # cv.imwrite(path, res)
    # cv.imshow('result', res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

if __name__ == '__main__':
    # create?MODNet?and?load?the?pre-trained?ckpt
    # cfg = None
    # cfg1 = None

    # json文件路径
    prune_info_path = "D:\\MODNet-ModelCompression\\result\\prune\\modify_modnet_photographic_portrait_matting\\modnet_p_new_trimap_0027_lr0.0001_ratio_0_thresh_1e-05.json"
    # 剪枝后模型路径
    ckp_pth = "D:\\MODNet-ModelCompression\\model_save\\modified_mppm\\new_trimap_0017_lr0.01.pth"
    prune_info = json.load(open(prune_info_path))
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
                         backbone_pretrained=False)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # modnet = modnet.to(device)
    modnet = torch.nn.DataParallel(modnet)
    # modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True,cfg=cfg,cfg1=cfg1))
    # ckp_pth = 'pretrained/modify_modnet_photographic_portrait_matting'
    # 模型文件路径
    # ckp_pth = "./pretrained/modify_modnet_photographic_portrait_matting"
    # ckp_pth = "./pretrained/our_modnet.ckpt"

    # ckp_pth = "./model_save/original_model_YOLO_4w_prue_MBV2_finetinue/new_trimap_0018_lr0.001.pth"
    # ckp_pth = "./model_save_soc/model_11M_0.0135/soc_0005.pth"
    if torch.cuda.is_available():
        modnet = modnet.cuda()

        weights = torch.load(ckp_pth)
    else:
        weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    # modnet.load_state_dict(weights)
    modnet.load_state_dict(weights, strict=True)

    pth = './src/datasets/PPM-100/val/fg'
    path_list = [os.path.join(pth,i) for i in os.listdir(pth)]

    total_time = 0.0
    num_images = len(path_list)

    # pth = './PPM-100/train/fg/49740440761_f9ffe43f60_o.jpg'
    # 修改
    # for i in path_list:
    #     infer_one_image(modnet, i)
    #至此
    # 正式计时推理
    for i, img_path in enumerate(path_list):
        # tic=time.time()
        elapsed = infer_one_image(modnet, img_path)
        # toc=time.time()
        # print(toc-tic)
        total_time += elapsed
        print(f"Processed image {i + 1}/{num_images}, time: {elapsed:.4f}s")

    # 计算FPS
    fps = num_images / total_time
    print(f"\n{'=' * 50}")
    print(f"Total images processed: {num_images}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    print(f"{'=' * 50}")
    # img = Image.open(pth)

    # matte = predit_matte(modnet, img)
    # prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
    # prd_img.save('test_predic.jpg')



