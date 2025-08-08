import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum

class SplitCriteria(Enum):
    INFOGAINRATIO = 1
    INFOGAIN = 2
    GINIINDEX = 3
    MAXOPERATOR = 4

def process():
    # 读取数据
    data = pd.read_csv('data/adult/adult_handle.csv')

    # 划分训练集与测试集
    np.random.seed(0)
    feat_names = data.columns[1:]
    label_name = data.columns[0]
    # 重排下标之后，按新的下标索引数据
    data = data.reindex(np.random.permutation(data.index))
    ratio = 0.8
    split = int(ratio * len(data))
    train_x = data[:split].drop(columns=['income']).to_numpy()
    train_y = data['income'][:split].to_numpy()
    test_x = data[split:].drop(columns=['income']).to_numpy()
    test_y = data['income'][split:].to_numpy()
    print('训练集大小：', len(train_x))
    print('测试集大小：', len(test_x))
    print('特征数：', train_x.shape[1])

    feat_ranges = {}
     # 遍历每个特征
    for i, feat_name in enumerate(feat_names):
        # 获取该特征在训练集中的所有唯一取值
        unique_values = np.unique(train_x[:, i])
        
        # 存储为排序后的列表
        feat_ranges[feat_name] = sorted(unique_values.tolist())

    return train_x, train_y, test_x, test_y, feat_ranges, feat_names
