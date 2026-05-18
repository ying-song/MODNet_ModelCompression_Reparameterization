import os
import subprocess
import re
from datetime import datetime
import argparse

def run_evaluation(ckpt_path, json_path):
    """
    运行评估命令并返回MSE和MAD值
    """
    cmd = f"CUDA_VISIBLE_DEVICES=0 python ./eval.py --ckpt-path {ckpt_path} --prune-info {json_path}"
    try:
        print(f"执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 从输出中提取MSE和MAD值
        mse_match = re.search(r'mse:\s*([\d.]+)', result.stdout.lower())
        mad_match = re.search(r'mad:\s*([\d.]+)', result.stdout.lower())
        
        if mse_match and mad_match:
            mse = float(mse_match.group(1))
            mad = float(mad_match.group(1))
            return mse, mad
        else:
            print(f"无法从输出中解析MSE和MAD值: {result.stdout}")
            return None, None
    except Exception as e:
        print(f"运行评估时出错: {str(e)}")
        return None, None

def read_existing_results(file_path):
    """
    读取现有的表格数据和头部信息
    """
    data = []
    header_info = None
    try:
        if not os.path.exists(file_path):
            return data, header_info
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 存储头部信息
            header_info = []
            for line in lines:
                if 'Training Results' in line or 'Training Completed' in line or 'Total training time' in line:
                    header_info.append(line.strip())
            
            for line in lines:
                if '-' in line and '|' not in line:
                    continue
                if 'Epoch' in line:
                    continue
                if '|' in line:
                    parts = [part.strip() for part in line.split('|')[1:-1]]
                    if len(parts) >= 2:  # 确保至少有基本的列
                        try:
                            epoch = int(parts[0])
                            row_data = {
                                'epoch': epoch,
                                'mse': float(parts[1]) if len(parts) > 1 else None,
                                'mad': float(parts[2]) if len(parts) > 2 else None
                            }
                            data.append(row_data)
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
    return data, header_info

def write_results_table(file_path, results, header_info):
    """
    写入评估结果表格
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入头部信息
            if header_info:
                for line in header_info:
                    f.write(f"{line}\n")
            else:
                # 如果没有头部信息，添加默认的评估时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Evaluation Results - {current_time}\n")
            
            # 写入表头分隔线
            f.write("-" * 80 + "\n")
            # 写入表头
            f.write("|  Epoch   |         MSE          |         MAD          |\n")
            # 写入表头分隔线
            f.write("-" * 80 + "\n")
            
            # 写入数据行
            for i, row in enumerate(results):
                epoch = row['epoch']
                mse = row.get('mse')
                mad = row.get('mad')

                f.write(f"|    {epoch:<5} |")
                
                if mse is not None:
                    f.write(f"       {mse:<11.6f}    |")
                else:
                    f.write(f"                      |")
                    
                if mad is not None:
                    f.write(f"       {mad:<11.6f}    |")
                else:
                    f.write(f"                      |")
                    
                f.write("\n")
                
                # 每五个epoch插入一个横向分隔符
                if (i + 1) % 5 == 0:
                    f.write("-" * 80 + "\n")
            
        print(f"成功更新表格，共处理 {len(results)} 行数据")
    except Exception as e:
        print(f"写入表格时出错: {str(e)}")

def extract_epoch_number(filename):
    """
    从模型文件名中提取epoch数
    支持格式：
    - model_epoch_001.ckpt
    - model_001.ckpt
    """
    # 尝试匹配 "epoch_XXX" 格式
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # 尝试匹配文件名中的数字部分
    match = re.search(r'(\d+)\.ckpt$', filename)
    if match:
        return int(match.group(1))
    
    return None

def main():
    parser = argparse.ArgumentParser(description="CKPT模型评估脚本")
    parser.add_argument("--model-dir", type=str, default='./result/prune/modify_modnet_photographic_portrait_matting/', help="包含ckpt模型文件的目录路径")
    parser.add_argument("--json-path", type=str, default='./result/prune/modify_modnet_photographic_portrait_matting/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.5.json', help="评估用的json配置文件路径")
    parser.add_argument("--output", type=str, default="./training_results/20241129_200722.txt", help="评估结果输出文件路径")
    args = parser.parse_args()

    # 检查目录和文件是否存在
    if not os.path.exists(args.model_dir):
        print(f"错误：模型目录不存在: {args.model_dir}")
        return
    
    if not os.path.exists(args.json_path):
        print(f"错误：JSON配置文件不存在: {args.json_path}")
        return

    # 获取所有ckpt文件并按epoch排序
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.ckpt')]
    model_files.sort(key=lambda x: extract_epoch_number(x) or 0)

    if not model_files:
        print(f"未在目录中找到ckpt文件: {args.model_dir}")
        return

    print(f"找到 {len(model_files)} 个ckpt文件")
    print("开始评估...")
    print("-" * 80)

    # 读取现有结果（如果有的话）
    existing_results, header_info = read_existing_results(args.output)
    results = []

    for model_file in model_files:
        epoch = extract_epoch_number(model_file)
        if epoch is None:
            print(f"跳过文件 {model_file} - 无法提取epoch编号")
            continue

        model_path = os.path.join(args.model_dir, model_file)
        print(f"正在评估 Epoch {epoch}...")
        print(f"模型文件: {model_file}")

        # 运行评估
        mse, mad = run_evaluation(model_path, args.json_path)
        
        if mse is not None and mad is not None:
            results.append({
                'epoch': epoch,
                'mse': mse,
                'mad': mad
            })
            print(f"Epoch {epoch} 评估完成: MSE={mse:.6f}, MAD={mad:.6f}")
        else:
            print(f"Epoch {epoch} 评估失败")
        
        print("-" * 80)

    # 写入结果
    if results:
        write_results_table(args.output, results, header_info)
        print(f"评估结果已保存到: {args.output}")
    else:
        print("没有成功评估任何模型")

if __name__ == "__main__":
    main()