import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os

# 创建目录保存树图
os.makedirs('trees', exist_ok=True)

# 加载Adult数据集
url = "data/adult/adult_100.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)

# 数据预处理
data = data.dropna()

# 特征和目标分离
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# 特征类型划分
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# 预处理流水线
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 预处理数据
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 获取特征名称
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)

# 随机森林模型
rf = RandomForestClassifier(
    n_estimators=1,        # 树的数量
    max_depth=2,           # 树的最大深度
    min_samples_split=2,    # 允许更细分裂
    min_samples_leaf=1,     # 允许单样本叶子
    max_features=0.8,       # 增加特征使用比例
    bootstrap=False,        # 禁用bootstrap保证用全数据
    random_state=42
)

# 训练模型
rf.fit(X_train_processed, y_train)

# 可视化每棵树（使用matplotlib替代graphviz）
for i, tree_in_forest in enumerate(rf.estimators_):
    plt.figure(figsize=(20, 10))
    
    # 使用plot_tree可视化
    plot_tree(
        tree_in_forest,
        feature_names=feature_names,
        class_names=['<=50K', '>50K'],
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=10
    )
    
    # 保存图像
    plt.savefig(f'picture/tree_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存树 {i+1} 的可视化图像: trees/tree_{i+1}.png")

# 预测与评估
y_pred = rf.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)

print(f"随机森林准确率: {accuracy:.4f}")
