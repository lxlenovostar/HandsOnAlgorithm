import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-4, reg_strength=0.1, verbose=False):
        """
        初始化逻辑回归模型
        
        参数:
        learning_rate -- 学习率（默认: 0.01）
        max_iters -- 最大迭代次数（默认: 1000）
        tol -- 收敛阈值（默认: 1e-4）
        reg_strength -- L2正则化强度（默认: 0.1）
        verbose -- 是否显示训练进度（默认: False）
        """
        self.lr = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.reg_strength = reg_strength
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    @staticmethod
    def sigmoid(z):
        """
        Sigmoid激活函数
        
        参数:
        z -- 输入值
        
        返回:
        sigmoid(z)的值，范围在(0,1)之间
        """
        # 为避免数值溢出，对大负值和小正值进行裁剪
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def log_loss(self, y_true, y_pred):
        """
        对数损失函数（交叉熵损失）
        
        参数:
        y_true -- 真实标签 (n_samples,)
        y_pred -- 预测概率 (n_samples,)
        
        返回:
        平均对数损失
        """
        # 避免log(0)的问题
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # 计算交叉熵损失
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # 添加L2正则化项
        if self.weights is not None:
            l2_penalty = 0.5 * self.reg_strength * np.sum(self.weights ** 2)
            loss += l2_penalty
            
        return loss
    
    def fit(self, X, y):
        """
        训练逻辑回归模型
        
        参数:
        X -- 训练数据 (n_samples, n_features)
        y -- 训练标签 (n_samples,)
        """
        # 确保y是1维向量
        if y.ndim > 1:
            y = y.flatten()
        
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 转换y为浮点型
        y = y.astype(float)
        
        # 梯度下降优化
        for i in range(self.max_iters):
            # 计算预测值和损失
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # 计算当前损失
            current_loss = self.log_loss(y, y_pred)
            self.loss_history.append(current_loss)
            
            # 检查收敛性
            if i > 0 and np.abs(self.loss_history[-2] - current_loss) < self.tol:
                if self.verbose:
                    print(f"在第 {i} 次迭代后收敛")
                break
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.reg_strength * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 打印进度
            if self.verbose and (i % 100 == 0 or i == 0):
                print(f"迭代 {i}/{self.max_iters}, 损失: {current_loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率值
        
        参数:
        X -- 输入数据 (n_samples, n_features)
        
        返回:
        预测概率 (n_samples,)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别标签
        
        参数:
        X -- 输入数据 (n_samples, n_features)
        threshold -- 分类阈值（默认: 0.5）
        
        返回:
        预测类别标签 (0或1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """
        计算模型在测试集上的准确率
        
        参数:
        X -- 测试数据 (n_samples, n_features)
        y -- 真实标签 (n_samples,)
        
        返回:
        模型准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 示例用法
if __name__ == "__main__":
    # 导入必要的库
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # 创建示例数据集
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2, 
        random_state=42
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 创建并训练模型
    model = LogisticRegression(
        learning_rate=0.1,
        max_iters=2000,
        reg_strength=0.01,
        verbose=True
    )
    model.fit(X_train, y_train)
    
    # 评估模型
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    # 查看最终权重
    print("模型权重:", model.weights)
    print("模型偏置:", model.bias)
    
    # 使用模型预测概率
    sample = X_test[0:1]  # 取第一个样本
    proba = model.predict_proba(sample)
    print(f"样本预测概率: {proba[0]:.4f}")
    
    # 损失曲线（可选绘图）
    if model.verbose:
        import matplotlib.pyplot as plt
        plt.plot(model.loss_history)
        plt.title('训练损失曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('对数损失')
        plt.show()