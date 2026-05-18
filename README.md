<div align="center">
  <a href="" target="_blank">
  <img width="50%" src="https://github.com/sisyphus-cv-lab/MODNet-ModelCompression/blob/7f2450fbd25af6b6f14a9537ecf2b41e2518d71f/data/logo.png"></a>
</div>
<div align="center">
  <p>
   <a href="https://img.shields.io/badge/Hello-Buddy~-red"><img src="https://img.shields.io/badge/Hello-Buddy~-red.svg"></a>
   <a href="https://img.shields.io/badge/Enjoy-Yourself-brightgreen"><img src="https://img.shields.io/badge/Enjoy-Yourself-brightgreen.svg"></a>
<br>
</div>

[简体中文](README.md) | [English](README.EN.md)

# 📣Introduction

基于**L1-Norm**评价准则，我们采用了一种**自适应与固定比例相结合**（Adaptive and Fixed-scale Pruning）的启发式剪枝策略对视频人像抠图模型MODNet进行压缩，该策略较大程度去除了MODNet中的冗余参数，并降低了计算代价，在存储资源的利用上节省了**79%**！🎆

此外，我们采用OpenVINO将边缘计算引入视频人像抠图技术，通过边缘端推理测试，剪枝模型**MODNet-P**取得了一定的速度提升与较好的视觉推理效果！🎆

🚩[论文地址](https://kns.cnki.net/kcms2/article/abstract?v=3uoqIhG8C44YLTlOAiTRKu87-SJxoEJu6LL9TJzd50kCWwlELj4QEt2QYEK5xJJCQa2NxbtW6xTd6f65Jx3C5xFcbi9VcIcw&uniplatform=NZKPT)

# ✨Usage

### 1 克隆项目并安装依赖

```bash
git clone https://github.com/sisyphus-cv-lab/MODNet-ModelCompression
cd MODNet-ModelCompression
pip install -r requirements.txt  
```

### 2 下载并解压数据集

```bash
wget -c https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -O src/datasets/PPM-100.zip
mkdir src/datasets
unzip src/datasets/PPM-100.zip -d src/datasets
```

### 3 下载模型

获取[地址](https://drive.google.com/drive/folders/1SiVFYBkrkokBdv-EGyz1UKjQebgvV2Wy?usp=share_link)，各模型含义如下：

* new_modnet：MODNet官方预训练模型。但为了便于剪枝，网络结构定义发生了轻微改变；
* new_mobilenetv2_human_seg：backbone官方预训练模型（改变同上）；
* our_modnet：通过在合成数据集上训练得到，作为剪枝的基准模型；
* pruned_modnet：剪枝与再训练后的模型；

### 4 模型剪枝 

```bash
python main_prune.py --ckpt_path .\pretrained\our_modnet.ckpt --ratio 0.5 --threshold 0.5
```

🔥NOTE：

* **python main.py -h** 获取剪枝时待输入的参数及相关介绍；
* threshold：用于控制MODNet主干网络MobileNetV2部分的剪枝阈值；
* ratio：用于控制MODNet其他网络分支的剪枝比例；
* 剪枝完成后，得到剪枝后的模型及其对应的网络配制文件（.json）；该配制文件用于<u>再训练、模型评估、模型推理以及模型导出</u>时网络的构建；

---

json文件信息如下：

```json
{
    "ratio": 模型剪枝比例,
    "threshold": 自适应分析阈值,
    "new_cfg": [
		MobileNetV2倒置残差块的输出通道数
    ],
    "new_expansion_cfg": [
        MobileNetV2倒置残差块中膨胀因子的大小
    ],
    "new_lr_channels": [
        LR分支的输出通道数
    ],
    "new_hr_channels": [
        HR分支的输出通道数
    ],
    "new_f_channels": [
        F分支的输出通道数
    ]
}
```

### 5 模型再训练

对步骤4中剪枝得到的模型进行再训练，恢复精度。

```bash
python .\src\trainer.py --model-path .\result\modnet_p_ratio_0.5_thresh_0.5.ckpt --batch-size 2 --epoch 4
```

🔥NOTE：默认每一个轮次保存模型，以便通过模型评价得到最佳模型；

### 6 模型评估 

```bash
python .\src\eval.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json
```

### 7 模型推理

```bash
python .\src\infer.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json
```

### 8 模型导出

将再训练后的最优剪枝模型导出，例如epoch0为最优模型，导出指令如下：

```bash
python .\onnx\export_onnx.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json --output-path .\result\modnet_p_ratio_0.5_thresh_0.5_best.onnx
```

### 9 模型优化

使用OpenVINO 中的模型优化器（model optimizer）融合BN层，从而实现模型的进一步压缩与加速。

```bash
mo --input_model .\result\modnet_p_ratio_0.5_thresh_0.5_best.onnx --model_name pruned_modnet --output_dir .\result\
```

### 10 MODNet-P 模型推理

得到模型优化得到xml与bin文件后，利用OpenVINO Python API 装载、完成模型推理。

```bash
python inference_openvino.py --model-path .\result\pruned_modnet.xml --image-path .\data\img.jpg --device CPU
```

# 🌞Results on PPM-100

### 剪枝前后模型对比

| 评估指标            | MODNet   | MODNet-P |
| :------------------ | :------- | :------- |
| 参数量/M            | 6.45     | 1.34     |
| 计算量/G            | 18.32    | 4.38     |
| 模型大小/M          | 25.64    | 5.66     |
| 均方误差（MSE）     | 0.009912 | 0.018713 |
| 平均绝对误差（MAD） | 0.013661 | 0.022816 |

🔥NOTE:

* 训练数据集通过随机组合得到，因此，表格中MODNet精度指标**MSE、MAD**与原论文不一致。

---

### 剪枝前后模型推理速度对比

| 硬件推理设备       | MODNet    | MODNet-P  |
| ------------------ | --------- | --------- |
| Intel i7-8565U CPU | 88.86 ms  | 45.93 ms  |
| NSC2               | 167.93 ms | 101.93 ms |
| ...                | ...       | ...       |

🔥NOTE:

* 使用OpenVINO在NSC2上推理时，需要采用**USB3.0**接口；

---

### 模型再训练方式对比

|                    | 微调     | 从头训练 |
| ------------------ | -------- | -------- |
| 固定主干网络剪枝   | 0.018291 | 0.015588 |
| 对整个模型进行剪枝 | 0.018632 | 0.016826 |

🔥NOTE:

* 进一步对比**微调**与**从头训练**两种方式的性能，固定主干网络与否对MODNet进行剪枝、测试；

* 为了便于观察比较，这里仅使用MSE作为评价准则。

# 📞Contact

关于本项目任何的疑问、建议，欢迎[submit issue](https://github.com/sisyphus-cv-lab/MODNet-ModelCompression/issues)或联系 hbchenstu@outlook.com.

# 📘Reference

https://github.com/ZHKKKe/MODNet

https://github.com/actboy/MODNet

https://github.com/Eric-mingjie/rethinking-network-pruning

https://github.com/kingpeter2015/libovmatting
