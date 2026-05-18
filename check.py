import os
from PIL import Image
import paramiko

# 检查本地图片文件是否损毁
def find_corrupted_images(local_folder_path):
    corrupted_images = []
    for filename in os.listdir(local_folder_path):
        img_path = os.path.join(local_folder_path, filename)
        try:
            with Image.open(img_path) as img:
                img.verify()  # 验证图片是否有效
        except (IOError, SyntaxError):
            corrupted_images.append(filename)
            print(f"Corrupted image found: {filename}")
    return corrupted_images

# 从远程服务器下载损毁的图片文件
def download_images_from_remote(remote_host, remote_port, username, password, remote_folder_path, local_folder_path, corrupted_images):
    # 使用 paramiko 连接到远程计算机
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_host, port=remote_port, username=username, password=password)

    sftp = ssh.open_sftp()
    for image_name in corrupted_images:
        # 生成远程路径并替换 Windows 的反斜杠为 Linux 正斜杠
        remote_image_path = os.path.join(remote_folder_path, image_name).replace('\\', '/')
        local_image_path = os.path.join(local_folder_path, image_name)

        try:
            # 检查远程文件是否存在
            sftp.stat(remote_image_path)
            sftp.get(remote_image_path, local_image_path)
            print(f"Downloaded {image_name} from remote server.")
        except FileNotFoundError:
            print(f"File {remote_image_path} not found on remote server.")
        except Exception as e:
            print(f"Failed to download {image_name} due to error: {e}")

    sftp.close()
    ssh.close()

# 配置本地和远程路径
local_folder_path = r'D:\MODNet-ModelCompression\src\datasets\new_data\train\alpha'          # 本地文件夹路径，使用 Windows 风格路径
remote_folder_path = '/home/xxfs/MODNet-ModelCompression/src/datasets/new_data/train/alpha'           # 远程文件夹路径，使用 Linux 风格路径
remote_host = '47.102.47.84'                          # 远程主机IP
remote_port = 8000                                       # 远程主机端口
username = 'xxfs'                                   # 用户名
password = 'xixifusi2024.'                                   # 密码

# 检查损毁图片并从远程服务器下载
corrupted_images = find_corrupted_images(local_folder_path)
if corrupted_images:
    download_images_from_remote(remote_host, remote_port, username, password, remote_folder_path, local_folder_path, corrupted_images)
else:
    print("No corrupted images found.")



