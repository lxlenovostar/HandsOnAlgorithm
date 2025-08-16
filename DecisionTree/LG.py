import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 加载Adult数据集
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url = "data/adult/adult_100.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)

# 数据预处理
print(f"原始数据量: {len(data)}")
data = data.dropna()
print(f"清洗后数据量: {len(data)}")

# 特征和目标分离
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)  # 二值编码

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 预处理数据
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 创建逻辑回归模型
model = LogisticRegression(
    penalty='l2',           # L2正则化
    C=0.1,                  # 正则化强度
    solver='lbfgs',         # 优化算法
    max_iter=1000,          # 最大迭代次数
    class_weight='balanced', # 处理类别不平衡
    random_state=42
)

# 训练模型
model.fit(X_train_processed, y_train)

# 预测与评估
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)

print(f"逻辑回归准确率: {accuracy:.4f}")
print(f"预测准确率: {accuracy*100:.2f}%")
