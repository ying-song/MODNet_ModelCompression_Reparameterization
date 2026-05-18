# Introduction

本项目针对 MODNet 人像抠图模型在移动端部署时计算量大的问题，提出了一种结合 结构化剪枝 与 结构重参数化 的轻量化优化方法。

剪枝部分基于论文《基于模型剪枝的轻量化人像抠图研究》，对 MODNet 的三个分支（语义、细节、融合）采用差异化剪枝策略，显著降低模型参数量和计算量。

针对剪枝后精度下降的问题，提出 PartialBNorm 结构，解耦 IBNorm 中的 BN 与 IN，实现重参数化融合，恢复模型精度。

最终模型在 PPM-100 数据集上，相比剪枝后模型 MSE 降低 **34.6%**，MAD 降低 **33.4%**。

提供基于 Django 的完整人像抠图 Web 系统，支持图片上传、模型选择、抠图预览、背景替换、历史记录管理等实用功能。



# Results on PPM-100

### 剪枝前后模型对比

| 评估指标            | 剪枝前   | 剪枝后 | 剪枝后所有分支重参数化 |
| :------------------ | :------- | :------- | :------------------ |
| 均方误差（MSE）     | 0.022979 | 0.047389 |      0.031018     |
| 平均绝对误差（MAD） | 0.026985 | 0.054365 |      0.036211     |
| 参数量/M            | 6.45     | 4.17     |      4.17     |
| 计算量/G            | 17.69    | 8.77     |      8.91    |
| 模型大小/M          | 26.3    | 11.9    |      12.0     |

🔥NOTE:

* 训练数据集通过随机组合得到，因此，表格中MODNet精度指标**MSE、MAD**与原论文不一致。

---


# 📘Reference

https://github.com/sisyphus-cv-lab/MODNet-ModelCompression?tab=readme-ov-file

https://github.com/ZHKKKe/MODNet

https://github.com/actboy/MODNet

https://github.com/Eric-mingjie/rethinking-network-pruning

https://github.com/kingpeter2015/libovmatting
