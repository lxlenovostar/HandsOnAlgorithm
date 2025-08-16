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
    # 定义列名
    column_names = [
        'checking_account_status', 'duration_months', 'credit_history', 'purpose',
        'credit_amount', 'savings_account', 'employment_since', 'installment_rate',
        'personal_status_sex', 'other_debtors', 'present_residence', 'property',
        'age', 'other_installment_plans', 'housing', 'existing_credits', 'job',
        'dependents', 'telephone', 'foreign_worker', 'credit_risk'
    ]

    # 读取数据
    data = pd.read_csv('data/german/german.data', 
                      header=None, 
                      delimiter=' ', 
                      names=column_names)

    # 将credit_risk列移动到第一列
    cols = data.columns.tolist()
    cols.insert(0, cols.pop(cols.index('credit_risk')))
    data = data[cols]

    # ==== 分箱处理 ====
    # 定义需要分箱的连续特征
    continuous_features = ['duration_months', 'credit_amount', 'age', 
                          'installment_rate', 'present_residence', 
                          'existing_credits', 'dependents']
    
    # 分箱策略配置
    bin_strategies = {
        'duration_months': [0, 12, 24, 36, 48, 72],
        'credit_amount': [0, 1000, 5000, 10000, 20000],
        'age': [18, 25, 35, 45, 60, 75],
        'installment_rate': [1, 2, 3, 4],
        'present_residence': [1, 2, 3, 4],
        'existing_credits': [0, 1, 2, 3, 4],
        'dependents': [1, 2, 3]
    }
    
    # 分箱标签映射
    bin_labels = {
        'duration_months': ['超短期(<1年)', '短期(1-2年)', '中期(2-3年)', '长期(3-4年)', '超长期(>4年)'],
        'credit_amount': ['小额(<1000DM)', '中额(1000-5000DM)', '大额(5000-10000DM)', '巨额(>10000DM)'],
        'age': ['青年(<25)', '青壮年(25-35)', '中年(35-45)', '中老年(45-60)', '老年(>60)'],
        'installment_rate': ['低负担', '中负担', '高负担'],
        'present_residence': ['短期居住', '中期居住', '长期居住'],
        'existing_credits': ['无贷款', '少量贷款', '中等贷款', '多笔贷款'],
        'dependents': ['少量抚养', '多抚养']
    }
    
    # 执行分箱
    for feature in continuous_features:
        # 确保标签数量匹配
        n_bins = len(bin_strategies[feature]) - 1
        n_labels = len(bin_labels[feature])
        
        if n_bins != n_labels:
            bin_labels[feature] = bin_labels[feature][:n_bins]
            print(f"警告: 自动调整 {feature} 标签数量为 {n_bins}")
        
        # 分箱处理
        data[f'{feature}_bin'] = pd.cut(
            data[feature],
            bins=bin_strategies[feature],
            labels=bin_labels[feature],
            include_lowest=True
        )
    
    # ==== 分箱处理结束 ====
    
    # 查看数据集信息
    print("===== 数据集基本信息 =====")
    print(data.info())
    
    # 查看前5行数据（包含分箱结果）
    print("\n===== 前5行数据示例（含分箱特征）=====")
    print(data.head())
    
    # ==== 优化点：存储分箱后的特征值 ====
    feat_ranges = {}
    
    # 处理所有特征
    for feat in data.columns:
        # 对于分箱特征，直接存储分箱标签
        if feat.endswith('_bin'):
            # 获取分箱特征的所有可能标签
            bin_labels = data[feat].unique().tolist()
            feat_ranges[feat] = sorted(bin_labels)
        else:
            # 对于非分箱特征，存储唯一值
            unique_vals = data[feat].unique().tolist()
            feat_ranges[feat] = sorted(unique_vals)
    
    # ==== 特征编码 ====
    # 将分类特征转换为整数编码
    for feat in data.columns:
        # 对于分箱特征，使用自定义标签顺序
        if feat.endswith('_bin'):
            # 使用feat_ranges中存储的标签顺序
            data[feat] = pd.Categorical(data[feat], categories=feat_ranges[feat])
        else:
            # 其他特征转换为分类类型
            data[feat] = data[feat].astype('category')
        
        # 转换为整数编码
        data[feat] = data[feat].cat.codes
    
    # ==== 划分训练集与测试集 ====
    np.random.seed(0)
    feat_names = data.columns[1:]
    label_name = data.columns[0]
    
    # 重排下标
    data = data.reindex(np.random.permutation(data.index))
    ratio = 0.8
    split = int(ratio * len(data))
    
    # 提取训练集和测试集
    train_x = data.iloc[:split, 1:].values
    train_y = data.iloc[:split, 0].values
    test_x = data.iloc[split:, 1:].values
    test_y = data.iloc[split:, 0].values
    
    print('\n训练集大小：', len(train_x))
    print('测试集大小：', len(test_x))
    print('特征数：', train_x.shape[1])
    
    # 打印分箱特征值
    print("\n===== 分箱特征值存储 =====")
    for feat, values in feat_ranges.items():
        if feat.endswith('_bin'):
            print(f"{feat}: {values}")
    
    return train_x, train_y, test_x, test_y, feat_ranges, feat_names