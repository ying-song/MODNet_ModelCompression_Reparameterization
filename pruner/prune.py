import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_nums_of_keep_channels(sparsity, initial_channels):
    """
    The number of reserved channels is obtained according to a fixed ratio.
    @param sparsity: User-set pruning ratio
    @param initial_channels: The output channel of the network layer
    @return:
    """
    return int((1 - sparsity) * initial_channels) if initial_channels != 1 else 1


# def compute_weights(m, threshold):
#     """
#     Adaptive pruning is used to calculate the channels that meet the
#     requirements according to the threshold
#     """
#     assert threshold != 0, "threshold must > 0"

#     weight = m.abs().numpy()
#     L1_norm = np.sum(weight, axis=(1, 2, 3))
#     x = sorted(L1_norm, reverse=True)
#     normalized = (x - np.min(x)) / (np.max(x) - np.min(x))

#     idx = 0
#     flag = True
#     for filter_idx in range(len(normalized)):
#         if normalized[filter_idx] < threshold and flag:
#             idx = filter_idx
#             flag = False

#     nums_keep = None
#     per = (idx + 1) / len(normalized)
#     if int(per * len(normalized)) != 1:
#         nums_keep = _make_divisible(per * len(normalized), 8)
#     else:
#         nums_keep = 1
#     return nums_keep

# 修改
def compute_weights(m, threshold):
    """
    Adaptive pruning is used to calculate the channels that meet the
    requirements according to the threshold.
    """
    assert threshold > 0, "Threshold must be greater than 0"

    # 计算 L1 范数
    weight = m.abs().numpy()  # 假设 m 是一个张量
    if weight.ndim == 4:  # 确保是四维数组
        L1_norm = np.sum(weight, axis=(1, 2, 3))  # 在通道维度上计算 L1 范数
    elif weight.ndim == 1:
        L1_norm = np.array([np.sum(weight)])  # 处理一维数组情况
    else:
        raise ValueError(f"Unexpected weight dimension: {weight.ndim}")

    # 检查 L1_norm 的类型并打印调试信息
    print(f"L1_norm shape: {L1_norm.shape}")  # 仅用于调试
    print(f"L1_norm: {L1_norm}")  # 仅用于调试

    # 将 L1 范数排序
    x = sorted(L1_norm, reverse=True)

    # 归一化 L1 范数
    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))  # 确保不会出现 NaN

    # 查找符合阈值的通道
    idx = 0
    flag = True
    for filter_idx in range(len(normalized)):
        if normalized[filter_idx] < threshold and flag:
            idx = filter_idx
            flag = False

    # 计算保留的通道数
    per = (idx + 1) / len(normalized)
    nums_keep = max(1, int(np.ceil(per * len(normalized) / 8.0) * 8))  # 保证结果为8的倍数
    return nums_keep

# 至此