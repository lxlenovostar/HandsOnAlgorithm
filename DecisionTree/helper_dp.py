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
    #data = pd.read_csv('data/titanic/train.csv')
    #data = pd.read_csv('data/titanic/train_20.csv')
    data = pd.read_csv('data/titanic/train_100.csv')
    # 查看数据集信息和前5行具体内容，其中NaN代表数据缺失
    #print(data.info())
    #print(data[:5])

    # 删去编号、姓名、船票, Cabin 4列
    data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare'], inplace=True)

    feat_ranges = {}
    #cont_feat = ['Age', 'Fare'] # 连续特征
    cont_feat = [] # 连续特征
    bins = 10 # 分类点数

    for feat in cont_feat:
        # 数据集中存在缺省值nan，需要用np.nanmin和np.nanmax
        min_val = np.nanmin(data[feat]) 
        max_val = np.nanmax(data[feat])
        feat_ranges[feat] = np.linspace(min_val, max_val, bins).tolist()
        #print(feat, '：') # 查看分类点
        #for spt in feat_ranges[feat]:
        #    print(f'{spt:.4f}')

    # 只有有限取值的离散特征
    #cat_feat = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked'] 
    cat_feat = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked'] 
    for feat in cat_feat:
        data[feat] = data[feat].astype('category') # 数据格式转为分类格式
        #print(f'{feat}：{data[feat].cat.categories}') # 查看类别
        data[feat] = data[feat].cat.codes.to_list() # 将类别按顺序转换为整数
        ranges = list(set(data[feat]))
        ranges.sort()
        feat_ranges[feat] = ranges

    # 将所有缺省值替换为-1
    data.fillna(-1, inplace=True)
    for feat in feat_ranges.keys():
        feat_ranges[feat] = [-1] + feat_ranges[feat]

    # 划分训练集与测试集
    np.random.seed(0)
    feat_names = data.columns[1:]
    label_name = data.columns[0]
    # 重排下标之后，按新的下标索引数据
    data = data.reindex(np.random.permutation(data.index))
    ratio = 0.8
    split = int(ratio * len(data))
    train_x = data[:split].drop(columns=['Survived']).to_numpy()
    train_y = data['Survived'][:split].to_numpy()
    test_x = data[split:].drop(columns=['Survived']).to_numpy()
    test_y = data['Survived'][split:].to_numpy()
    print('训练集大小：', len(train_x))
    print('测试集大小：', len(test_x))
    print('特征数：', train_x.shape[1])

    return train_x, train_y, test_x, test_y, feat_ranges, feat_names
