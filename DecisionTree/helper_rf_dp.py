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
    cat_columns = [
    'class', 'cap.shape', 'cap.surface', 'cap.color', 'bruises', 'odor',
    'gill.attachment', 'gill.spacing', 'gill.size', 'gill.color',
    'stalk.shape', 'stalk.root', 'stalk.surface.above.ring', 
    'stalk.surface.below.ring', 'stalk.color.above.ring',
    'stalk.color.below.ring', 'veil.type', 'veil.color', 'ring.number',
    'ring.type', 'spore.print.color', 'population', 'habitat'
    ]

    # 读取数据
    data = pd.read_csv('data/mushroom/agaricus-lepiota.data', names = cat_columns)
    # 查看数据集信息和前5行具体内容，其中NaN代表数据缺失
    print(data.info())
    print(data[:5])

    """
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

    """
    # Mushroom数据集属性类型定义
    feat_status = {
        # 离散属性 (22个)
        'cap.shape': 'categorical',        # 菌盖形状: bell, conical, convex, flat, knobbed, sunken
        'cap.surface': 'categorical',      # 菌盖表面: fibrous, grooves, scaly, smooth
        'cap.color': 'categorical',        # 菌盖颜色: brown, buff, cinnamon, gray, green, pink, purple, red, white, yellow
        'bruises': 'categorical',          # 擦伤: bruises, no
        'odor': 'categorical',             # 气味: almond, anise, creosote, fishy, foul, musty, none, pungent, spicy
        'gill.attachment': 'categorical', # 菌褶附着: attached, descending, free, notched
        'gill.spacing': 'categorical',     # 菌褶间距: close, crowded, distant
        'gill.size': 'categorical',        # 菌褶大小: broad, narrow
        'gill.color': 'categorical',       # 菌褶颜色: black, brown, buff, chocolate, gray, green, orange, pink, purple, red, white, yellow
        'stalk.shape': 'categorical',      # 菌柄形状: enlarging, tapering
        'stalk.root': 'categorical',       # 菌柄根部: bulbous, club, cup, equal, rhizomorphs, rooted, missing
        'stalk.surface.above.ring': 'categorical', # 菌环上方表面: fibrous, scaly, silky, smooth
        'stalk.surface.below.ring': 'categorical', # 菌环下方表面: fibrous, scaly, silky, smooth
        'stalk.color.above.ring': 'categorical',   # 菌环上方颜色: brown, buff, cinnamon, gray, orange, pink, red, white, yellow
        'stalk.color.below.ring': 'categorical',   # 菌环下方颜色: brown, buff, cinnamon, gray, orange, pink, red, white, yellow
        'veil.type': 'categorical',        # 菌幕类型: partial, universal
        'veil.color': 'categorical',       # 菌幕颜色: brown, orange, white, yellow
        'ring.number': 'categorical',      # 菌环数量: none, one, two
        'ring.type': 'categorical',       # 菌环类型: cobwebby, evanescent, flaring, large, none, pendant, sheathing, zone
        'spore.print.color': 'categorical', # 孢子印颜色: black, brown, buff, chocolate, green, orange, purple, white, yellow
        'population': 'categorical',      # 种群: abundant, clustered, numerous, scattered, several, solitary
        'habitat': 'categorical',         # 栖息地: grasses, leaves, meadows, paths, urban, waste, woods
    }

    feat_ranges = {}
    # 只有有限取值的离散特征
    for feat in cat_columns:
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
    train_x = data[:split].drop(columns=['class']).to_numpy()
    train_y = data['class'][:split].to_numpy()
    test_x = data[split:].drop(columns=['class']).to_numpy()
    test_y = data['class'][split:].to_numpy()
    print('训练集大小：', len(train_x))
    print('测试集大小：', len(test_x))
    print('特征数：', train_x.shape[1])

    return train_x, train_y, test_x, test_y, feat_ranges, feat_names, feat_status
