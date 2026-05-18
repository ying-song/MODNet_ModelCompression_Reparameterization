import os
from glob import glob

def check_dataset(image_dir, matte_dir):
    image_path = os.path.join(image_dir, '*')
    matte_path = os.path.join(matte_dir, '*')

    image_file_name_list = glob(image_path)
    matte_file_name_list = glob(matte_path)

    unmatched_images = []
    unmatched_mattes = []

    for img_name in image_file_name_list:
        base_name = os.path.basename(img_name).replace('.jpg', '').replace('.png', '')  # 根据实际格式修改
        expected_matte_name = os.path.join(matte_dir, base_name + '.png')  # 假设是 .png 格式
        if expected_matte_name not in matte_file_name_list:
            unmatched_images.append(img_name)

    for matte_name in matte_file_name_list:
        base_name = os.path.basename(matte_name).replace('.png', '').replace('.jpg', '')  # 根据实际格式修改
        expected_image_name = os.path.join(image_dir, base_name + '.jpg')  # 假设是 .jpg 格式
        if expected_image_name not in image_file_name_list:
            unmatched_mattes.append(matte_name)

    # 输出不匹配的文件
    if unmatched_images:
        print("Unmatched Images:")
        for img in unmatched_images:
            print(img)

    if unmatched_mattes:
        print("Unmatched Mattes:")
        for matte in unmatched_mattes:
            print(matte)

if __name__ == '__main__':
    image_dir = '/home/xxfs/MODNet-ModelCompression/src/datasets/new_data/train/image'  # 修改为你的图像目录
    matte_dir = '/home/xxfs/MODNet-ModelCompression/src/datasets/new_data/train/alpha'  # 修改为你的 alpha 目录

    check_dataset(image_dir, matte_dir)
