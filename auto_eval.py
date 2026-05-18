import os
import subprocess
import re
from datetime import datetime
import argparse

def run_evaluation(ckpt_path, json_path):
    """
    运行评估命令并返回MSE和MAD值
    """
    cmd = f"CUDA_VISIBLE_DEVICES=1 python ./eval2.py --ckpt-path {ckpt_path} --prune-info {json_path}"
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
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
           # 创建一个列表来存储所有匹配的头部信息
            header_info = []

            # 读取头部信息
            for line in lines:
                # 检查关键词
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
                    if len(parts) >= 4:  # 确保至少有基本的列
                        try:
                            epoch = int(parts[0])
                            row_data = {
                                'epoch': epoch,
                                'semantic_loss': float(parts[1]),
                                'detail_loss': float(parts[2]),
                                'matte_loss': float(parts[3])
                            }
                            data.append(row_data)
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
    return data, header_info

def write_new_table(file_path, original_data, mse_mad_results, header_info):
    """
    写入新的表格，包含MSE和MAD列，并在每行添加分隔线，并在每5个epoch插入横向分隔符
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入原始头部信息
            if header_info:
                for line in header_info:
                    f.write(f"{line}\n")
            
            # 写入表头分隔线
            f.write("-" * 127 + "\n")
            # 写入表头
            f.write("|  Epoch   |    Semantic Loss     |     Detail Loss      |      Matte Loss      |         MSE          |         MAD          |\n")
            # 写入表头分隔线
            f.write("-" * 127 + "\n")
            
            # 写入数据行并在每五个epoch插入分隔符
            for i, row in enumerate(original_data):
                epoch = row['epoch']
                # 从mse_mad_results获取MSE和MAD值，使用epoch作为键
                mse, mad = mse_mad_results.get(epoch, (None, None))  # 此处不需要epoch-1

                f.write(f"|    {epoch:<5} |       {row['semantic_loss']:<11.6f}    |       {row['detail_loss']:<11.6f}    |")
                f.write(f"       {row['matte_loss']:<11.6f}    |")
                
                if mse is not None and mad is not None:
                    f.write(f"       {mse:<11.6f}    |       {mad:<11.6f}    |\n")
                else:
                    f.write(f"                      |                      |\n")
                
                # 每五个epoch插入一个横向分隔符
                if (i + 1) % 5 == 0:
                    f.write("-" * 127 + "\n")
            
        print(f"成功更新表格，共处理 {len(original_data)} 行数据")
    except Exception as e:
        print(f"写入新表格时出错: {str(e)}")

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
    if match:
        return int(match.group(1))
    return None

def main(json_path=None):
    # 配置路径
    model_dir = "./model_save/modify1layer_rep_m_p_n_t_0027_lr0.0001_r_0.5_t_0.5/"
    results_file = "./training_results/20241028_train.txt"
    
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
    
    for model_file in model_files:
        epoch = extract_epoch_number(model_file)
        if epoch is None:
            print(f"跳过文件 {model_file} - 无法提取epoch编号")
            continue

        model_path = os.path.join(model_dir, model_file)
        
        # 根据文件类型判断是否需要加1
        if model_file.endswith('.ckpt'):
            table_epoch = epoch + 1
        else:
            table_epoch = epoch  # 对于.pth文件直接使用epoch

        print(f"正在评估 Epoch {table_epoch}...")
        print(f"模型文件: {model_file}")
        
        # 运行评估
        mse, mad = run_evaluation(model_path, json_path)
        
        if mse is not None and mad is not None:
            mse_mad_results[epoch] = (mse, mad)
            print(f"Epoch {table_epoch} 评估完成: MSE={mse:.6f}, MAD={mad:.6f}")
        else:
            print(f"Epoch {table_epoch} 评估失败")
        
        print("-" * 80)

    
    # 写入新的表格
    write_new_table(results_file, original_data, mse_mad_results, header_info)
    print("表格更新完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行评估模型的脚本")
    parser.add_argument("--json-path", type=str, help="指定json文件路径", default='./model_save/m_p_n_t_0027_lr0.0001_r_0.5_t_0.5/modnet_p_new_trimap_0027_lr0.0001_ratio_0.5_thresh_0.5.json', required=False)
    args = parser.parse_args()

    main(json_path=args.json_path)

