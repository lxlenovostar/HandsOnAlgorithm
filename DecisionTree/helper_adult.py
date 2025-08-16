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
    #data = pd.read_csv('data/adult/adult_handle_100.csv')
    #data = pd.read_csv('data/adult/adult_handle_1000.csv')

    # 根据Adult数据集官方文档定义特征类型
    feat_status = {
        # 连续属性 (6个)
        'age': 'continuous',
        'fnlwgt': 'continuous',          # 人口普查权重
        'education_num': 'continuous',  # 受教育年限
        'capital_gain': 'continuous',    # 资本收益
        'capital_loss': 'continuous',    # 资本损失
        'hours_per_week': 'continuous',  # 每周工作时长
    
        # 离散属性 (8个)
        'workclass': 'categorical',      # 工作类型
        'education': 'categorical',      # 教育程度
        'marital_status': 'categorical', # 婚姻状况
        'occupation': 'categorical',     # 职业
        'relationship': 'categorical',   # 家庭关系
        'race': 'categorical',           # 种族
        'sex': 'categorical',            # 性别
        'native_country': 'categorical'  # 原籍国家
    }
    """
  
    feat_status = {
        # 连续属性 (6个)
        'age': 'continuous',
        'fnlwgt': 'continuous',          # 人口普查权重
        'education_num': 'continuous',  # 受教育年限
        'capital_gain': 'continuous',    # 资本收益
        'capital_loss': 'continuous',    # 资本损失
        'hours_per_week': 'continuous',  # 每周工作时长
    
        # 离散属性 (8个)
        'workclass': 'continuous',      # 工作类型
        'education': 'continuous',      # 教育程度
        'marital_status': 'continuous', # 婚姻状况
        'occupation': 'continuous',     # 职业
        'relationship': 'continuous',   # 家庭关系
        'race': 'continuous',           # 种族
        'sex': 'continuous',            # 性别
        'native_country': 'continuous'  # 原籍国家
    }
    """


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

    return train_x, train_y, test_x, test_y, feat_ranges, feat_names, feat_status
