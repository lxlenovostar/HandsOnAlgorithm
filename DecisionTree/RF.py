import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os

# 创建目录保存树图
os.makedirs('trees', exist_ok=True)

# 加载Adult数据集
url = "data/adult/adult.data"
#url = "data/adult/adult_1000.data"
#url = "data/adult/adult_100.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)

# 数据预处理
data = data.dropna()

# ==== 修改1: 使用LabelEncoder处理目标变量 ====
y = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# 特征和目标分离
X = data.drop('income', axis=1)

# 特征类型划分
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# ==== 修改2: 去除独热编码，使用LabelEncoder处理分类特征 ====
# 创建标签编码器字典
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le  # 保存编码器用于后续解释

# 预处理流水线（仅标准化数值特征）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)
    ],
    remainder='passthrough'  # 保留已编码的分类特征
)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 预处理数据
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==== 修改3: 获取特征名称 ====
# 创建特征名称列表
feature_names = numerical_cols + categorical_cols

# 随机森林模型
rf = RandomForestClassifier(
    n_estimators=25,        # 树的数量
    max_depth=5,            # 树的最大深度
    min_samples_split=2,    # 允许更细分裂
    min_samples_leaf=1,     # 允许单样本叶子
    max_features=0.8,       # 增加特征使用比例
    bootstrap=False,        # 禁用bootstrap保证用全数据
    random_state=42
)

# 训练模型
rf.fit(X_train_processed, y_train)

# 初始化叶子节点计数器
leaf_counts = []

# ==== 新增功能：连续属性分裂点分析 ====
print("\n===== 连续属性分裂点分析 =====")

# 遍历每棵树
for i, tree_in_forest in enumerate(rf.estimators_):
    tree = tree_in_forest.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    
    # 存储当前树的连续属性分裂点
    cont_splits = {}
    
    # 遍历所有节点
    for node_id in range(n_nodes):
        # 跳过叶节点（无分裂）
        if children_left[node_id] == children_right[node_id]:
            continue
            
        # 获取特征索引和名称
        feat_idx = feature[node_id]
        if feat_idx < 0:  # 无效特征索引
            continue
            
        feat_name = feature_names[feat_idx]
        
        # 检查是否为连续属性
        if feat_name in numerical_cols:
            split_point = threshold[node_id]
            
            # 记录分裂点
            if feat_name not in cont_splits:
                cont_splits[feat_name] = []
            cont_splits[feat_name].append(split_point)
    
    # 打印当前树的连续属性分裂点
    #print(f"\n树 {i+1} 的连续属性分裂点:")
    #for feat, splits in cont_splits.items():
    #    print(f"  - {feat}: {splits}")
    
    # ==== 新增：反标准化分裂点 ====
    scaler = preprocessor.named_transformers_['num']
    original_splits = {}
    
    for feat, splits in cont_splits.items():
        # 获取特征在数值特征中的索引
        feat_idx_in_num = numerical_cols.index(feat)
        
        # 反标准化：原始值 = 标准化值 * 标准差 + 均值
        mean = scaler.mean_[feat_idx_in_num]
        std = scaler.scale_[feat_idx_in_num]
        original_values = [val * std + mean for val in splits]
        
        original_splits[feat] = original_values
    
    #print(f"\n树 {i+1} 的原始连续属性分裂点:")
    #for feat, splits in original_splits.items():
    #    print(f"  - {feat}: {[round(x, 2) for x in splits]}")
    

# 初始化叶子节点计数器
leaf_counts = []

# 可视化每棵树
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
    #plt.savefig(f'picture/tree_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存树 {i+1} 的可视化图像: trees/tree_{i+1}.png")

    # 计算并记录叶子节点数
    n_leaves = tree_in_forest.tree_.n_leaves
    leaf_counts.append(n_leaves)
    print(f"树 {i+1} 的叶子节点数: {n_leaves}")

# 打印叶子节点统计信息
print("\n===== 叶子节点统计 =====")
print(f"总树数: {len(leaf_counts)}")
print(f"叶子节点总数: {sum(leaf_counts)}")
print(f"平均每棵树叶子节点数: {sum(leaf_counts)/len(leaf_counts):.2f}")
print(f"最大叶子节点数: {max(leaf_counts)}")
print(f"最小叶子节点数: {min(leaf_counts)}")

# ==== 新增：计算训练集准确率 ====
y_train_pred = rf.predict(X_train_processed)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n训练集准确率: {train_accuracy:.4f}")

# ==== 新增：计算测试集准确率 ====
y_test_pred = rf.predict(X_test_processed)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f}")


# ==== 修改4: 添加特征重要性分析 ====
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

#print("\n===== 特征重要性排序 =====")
#for f in range(X_train_processed.shape[1]):
#    print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")