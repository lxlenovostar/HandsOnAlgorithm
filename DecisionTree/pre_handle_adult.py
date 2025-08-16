import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# 定义列名
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# 加载数据
data = pd.read_csv(
    'data/adult/adult.data',
    header=None,
    names=column_names,
    na_values='?',
    skipinitialspace=True
)

# 1. 处理缺失值
print("原始缺失值统计:")
print(data.isnull().sum())

# 删除包含缺失值的行
data_clean = data.dropna().reset_index(drop=True)

print("\n清洗后数据形状:", data_clean.shape)

# 2. 目标变量编码并移到第一列
data_clean['income'] = data_clean['income'].apply(
    lambda x: 1 if '>50K' in x else 0
)

# 3. 连续特征分箱处理（10类）
continuous_features = [] 
#continuous_features = ['age', 'fnlwgt', 'education_num', 
                      # 'capital_gain', 'capital_loss', 'hours_per_week']

# 创建分箱器
discretizer = KBinsDiscretizer(
    n_bins=10, 
    encode='ordinal',  # 输出整数编码
    strategy='quantile'  # 等频分箱
)

# 应用分箱
data_discrete = data_clean.copy()
#data_discrete[continuous_features] = discretizer.fit_transform(
#    data_clean[continuous_features]
#).astype(int)

# 4. 离散特征整数编码
categorical_features = ['workclass', 'education', 'marital_status',
                        'occupation', 'relationship', 'race', 
                        'sex', 'native_country']

# 创建编码字典
encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    data_discrete[feature] = le.fit_transform(data_clean[feature])
    encoders[feature] = le  # 保存编码器供后续使用

# 6. 将标签列移到第一列
cols = data_discrete.columns.tolist()
cols.insert(0, cols.pop(cols.index('income')))
data_discrete = data_discrete[cols]

# 7. 数据预览
print("\n预处理后数据示例:")
print(data_discrete.head())
print("\n数据类型统计:")
print(data_discrete.dtypes)

# 8. 保存处理后的数据
data_discrete.to_csv('data/adult/adult_handle.csv', index=False)
print("\n数据已保存至: data/adult/adult_processed.csv")