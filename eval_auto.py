import os
import subprocess
import re
from datetime import datetime
import argparse
import time

def run_evaluation(ckpt_path, json_path):
    """运行评估命令并返回指标和耗时"""
    cmd = f"CUDA_VISIBLE_DEVICES=0 python ./eval2.py --ckpt-path {ckpt_path} --prune-info {json_path}"
    try:
        print(f"执行命令: {cmd}")
        start_time = time.time()  # 记录开始时间

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # 计算耗时
        inference_time = time.time() - start_time

        # 解析输出结果
        metrics = {
            'mse': None,
            'mad': None,
            'inference_time': inference_time  # 添加全过程耗时
        }

        # 解析指标
        mse_match = re.search(r'mse:\s*([\d.]+)', result.stdout.lower())
        mad_match = re.search(r'mad:\s*([\d.]+)', result.stdout.lower())
        time_match = re.search(r'inference time:\s*([\d.]+)\s*s', result.stdout.lower())

        if mse_match and mad_match:
            metrics.update({
                'mse': float(mse_match.group(1)),
                'mad': float(mad_match.group(1))
            })

        if time_match:
            metrics['inference_time'] = float(time_match.group(1))

        return metrics
    except Exception as e:
        print(f"运行评估时出错: {str(e)}")
        return None


def read_existing_results(file_path):
    """
    读取现有的表格数据和头部信息
    """
    data = []
    header_info = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

           # 创建一个列表来存储所有匹配的头部信息
            header_info = []

            # 读取头部信息
            for line in lines:
                # 检查关键词
                # if 'SOC Fine-tuning Results' in line or 'SOC Fine-tuning Completed' in line or 'Total fine-tuning time' in line:
                if 'Training Results' in line or 'Training Completed' in line or 'Total training time' in line:
                    # 去掉首尾空白并添加到列表
                    header_info.append(line.strip())


            for line in lines:
                # 跳过分隔线
                if '-' in line and '|' not in line:
                    continue
                # 跳过表头
                if 'Epoch' in line:
                    continue
                # 处理数据行
                if '|' in line:
                    parts = [part.strip() for part in line.split('|')[1:-1]]  # 去掉首尾空元素
                    if len(parts) >= 3:  # 确保至少有基本的列
                        try:
                            epoch = int(parts[0])
                            row_data = {
                                'epoch': epoch,
                                'semantic_loss': float(parts[1]),
                                'detail_loss': float(parts[2]),
                                'matte_loss': float(parts[3])
                                # 'SOC Semantic Loss': float(parts[1]),
                                # 'SOC Detail Loss': float(parts[2]),
                            }
                            data.append(row_data)
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
    return data, header_info


def write_new_table(file_path, original_data, eval_results, header_info):
    """生成包含全过程耗时的新表格"""
    column_widths = [6, 16, 16, 16, 16, 16, 16]  # 各列宽度
    separator = '-' * (sum(column_widths) + (len(column_widths) - 1) * 3 + 6)
    header = (
        "|  Epoch   |    Semantic Loss  |    Detail Loss   |    Matte Loss    |"
        "       MSE        |       MAD      |  Inference Time   |"
    )

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write('\n'.join(header_info) + '\n')
            f.write(separator + '\n')
            f.write(header + '\n')
            f.write(separator + '\n')

            # 写入数据行
            for idx, row in enumerate(original_data):
                epoch = row['epoch']
                res = eval_results.get(epoch, {})

                # 构建数据行
                line = (
                    f"| {epoch:^8} | "
                    f"{row['semantic_loss']:^17.6f} | "
                    f"{row['detail_loss']:^16.6f} | "
                    f"{row['matte_loss']:^16.6f} | "
                    f"{(res.get('mse') or 0):^16.6f} | "
                    f"{(res.get('mad') or 0):^14.6f} | "
                    f"{(res.get('inference_time') or 0):^17.2f} |"
                )

                f.write(line + '\n')

                # 每5个epoch加分隔线
                if (idx + 1) % 5 == 0:
                    f.write(separator + '\n')

        print(f"成功生成包含全过程耗时的表格，共{len(original_data)}行")
    except Exception as e:
        print(f"写入表格失败: {str(e)}")

def get_base_filename(filename):
    """
    获取模型文件对应的基础文件名（用于匹配json文件）
    """
    return re.sub(r'_epoch\d+\.(ckpt|pth)$', '', filename)

def extract_epoch_number(filename):
    """
    从模型文件名中提取epoch数，例如从 'new_trimap_0001_lr0.01.pth' 提取 '0001'
    """
    match = re.search(r'new_trimap_(\d+)_lr', filename)
    # match = re.search(r'soc_auto_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main(json_path=None):
    # 配置路径
    model_dir = "./model_save/pruned_once_rep_lr_all5x5layers"
    results_file = "./training_results/20250809_215326.txt"

    # 读取现有的表格数据和头部信息
    original_data, header_info = read_existing_results(results_file)
    if not original_data:
        print("警告：未能读取到现有数据！")
        return

    print(f"成功读取到 {len(original_data)} 行现有数据")

    # 获取所有pth模型文件并按epoch轮数排序
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    model_files.sort(key=lambda x: extract_epoch_number(x) or 0)

    if not model_files:
        print("未找到pth模型文件！")
        return

    # 如果没有指定json路径，则使用默认路径
    if not json_path:
        base_name = get_base_filename(model_files[0])
        json_file = f"{base_name}.json"
        json_path = os.path.join(model_dir, json_file)

    if not os.path.exists(json_path):
        print(f"未找到对应的json文件: {json_path}")
        return

    print("开始评估模型...")
    print(f"使用json文件: {json_path}")
    print("-" * 80)

    # 存储MSE和MAD结果
    mse_mad_results = {}

    # 评估流程
    eval_results = {}
    for model_file in model_files:
        epoch = extract_epoch_number(model_file)
        if not epoch:
            continue

        metrics = run_evaluation(
            os.path.join(model_dir, model_file),
            json_path or get_base_filename(model_file)
        )

        if metrics:
            eval_results[epoch] = metrics
            print(f"Epoch {epoch} 评估结果:")
            print(f"  → MSE: {metrics['mse']:.6f}")
            print(f"  → MAD: {metrics['mad']:.6f}")
            print(f"  → 推理耗时: {metrics['inference_time']:.2f}s")
        else:
            print(f"Epoch {epoch} 评估失败")

        print("-" * 80)

    # 写入表格
    write_new_table(results_file, original_data, eval_results, header_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行评估模型的脚本")
    parser.add_argument("--json-path", type=str, help="指定json文件路径", default='./result/prune/modify_mppm_prune_once/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.4.json', required=False)
    args = parser.parse_args()

    main(json_path=args.json_path)