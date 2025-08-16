import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 
import random
from graphviz import Digraph
from typing import List, Union


class Node:
    def __init__(self):
        # 内部结点的feat表示用来分类的特征
        self.feat = '' 
        # 分类值列表，表示按照其中的值向子结点分类, key is value, value is index of self.child
        self.split = {} 
        # 子结点列表，叶结点的child为空
        self.child = []
        # 叶结点的label表示该结点对应的分类结果
        self.label = None
  
        # 逻辑回归扩展属性
        self.parent = None
        self.logistic_model = None   # sklearn模型对象
        self.feature_indices = []    # store feature index 
        self.is_logistic = False     # 节点类型标记
        #self.log_loss = None        # 训练损失
        #self.samples = 0            # 节点样本数


class DiffPRFM:
    # line 2 in Algorithm 2 Differential private ID3
    # Y for C, feat_names for A
    def __init__(self, inputs):

        X = inputs[0]
        Y = inputs[1]
        feat_names = inputs[2]
        feat_status = inputs[3]
        d = inputs[4]
        B = inputs[5]
        self.tree_id = inputs[6]

        self.root = Node()
        self.T = self.get_sample_dataset(X, Y, self.tree_id) 
        #self.T = np.hstack((X, Y.reshape(-1, 1)))
        self.feat_names = feat_names 
        self.feat_status = feat_status 
        self.d = d # d + 1 the depth of tree
        #self.B = B # differential privacy budget
        #self.e = self.B / (2*(d + 1))
        self.Leaf_B = 0.6*B
        self.B = 0.4*B # differential privacy budget
        self.e = self.B / (2*d)
        self.Leaf = 0 # 记录叶结点个数
        self.log_Leaf = 0

        self.log_feat_names = feat_names
        #self.log_T = self.T.copy()
        self.log_feat_index = self.get_feat_index()
        # 基于特征数量的动态阈值
        #self.regression_threshold = max(50, int(len(feat_names) * 3))
        self.regression_threshold = min(self.T.shape[0]/200, int(len(feat_names) * 3))
        
        np.random.seed(self.tree_id)
        #np.random.seed(int(time.time()) + self.tree_id)
        self.Build_DiffPID3(self.root, self.T, self.d, self.e, self.feat_names.tolist())

        print('number of tree leaf: ', self.Leaf)
        print('number of tree log_leaf: ', self.log_Leaf)
    
    def get_sample_dataset(self, X, Y, tree_id):
        # 新增随机采样逻辑
        #n_samples = X.shape[0]
        n_samples = int(X.shape[0]/20)
        rng = np.random.RandomState(int(time.time()) + tree_id)
        #indices = rng.choice(n_samples, size=n_samples, replace=True)  # 有放回抽样
        indices = rng.choice(n_samples, size=n_samples, replace=False)  # 有放回抽样
        X_sampled = X[indices]
        Y_sampled = Y[indices]

        return np.hstack((X_sampled, Y_sampled.reshape(-1, 1)))
    
    def get_feat_index(self):
        index_map = {}
        for idx, s in enumerate(self.log_feat_names):
            if s not in index_map:  # 只记录首次出现位置
                index_map[s] = idx
        return index_map

    def get_max_A(self, X):
        # 遍历每一列
        max_A_len = -1
        for i in range(X.shape[1]):
            # 获取当前列
            column = X[:, i]
            # 找出该列的唯一值
            unique_values_len = len(np.unique(column))
            if max_A_len < unique_values_len:
                max_A_len = unique_values_len
        return max_A_len
    
    def get_Noisy(self, e):
        # 添加拉普拉斯噪声
        # 这里假设敏感度为 1，隐私预算为 e 
        sensitivity = 1
        epsilon = e
        scale = sensitivity / epsilon

        noise = np.random.laplace(0, scale)
        return noise

        #bound = 3 * scale  # 3σ裁剪（覆盖99.7%分布）
        #bound = scale  # 3σ裁剪（覆盖99.7%分布）
        #print('what noise', noise, bound)
        #return np.clip(noise, -bound, bound)
    
    def get_heuristic_parameters(self, Nt, t, len_C, e):
        # line 9 in Algorithm Differential Private ID3
        if Nt / t*len_C < (2 ** 0.5) / e:
            return True
        else:
            return False
    
    def partition_C(self, Y):
        # 找到数组中的唯一值
        unique_values = np.unique(Y)

        # 按唯一值分割数组
        split_ndarrays = {}
        for value in unique_values:
            split_ndarrays[value] = Y[Y == value].reshape(-1, 1)

        #for value, sub_arr in split_ndarrays.items():
            #print(f"值为 {value} 的 ndarray：")
            #print(len(sub_arr))
            #print(type(split_arrays)) 

        return split_ndarrays 
    
    def partition_A(self, T, index):
        # TODO check index is valid
        # 找到数组中的唯一值
        unique_values = np.unique(T[:, index])

        # 用于存储切分结果的字典
        split_arrays = {}

        # 遍历每个唯一取值
        for value in unique_values:
            # 创建一个布尔掩码，筛选出该列取值等于当前值的行
            mask = T[:, index] == value
            # 根据掩码从原数组中选取满足条件的行
            split_arrays[value] = T[mask]

        # 打印切分结果
        #for value, sub_arr in split_arrays.items():
            #print(f"当第 {target_col_index} 列取值为 {value} 时，切分得到的子数组形状为: {sub_arr.shape}")

        return split_arrays 
    

    # 工具函数，计算 a * log a
    def aloga(self, a):
        return a * np.log2(a + 1e-8)

    # 计算某个子数据集的熵
    # Calculate entropy by category
    def entropy(self, Y):
        cnt = np.unique(Y, return_counts=True)[1] # 统计每个类别出现的次数
        N = len(Y)
        ent = -np.sum([self.aloga(Ni / N) for Ni in cnt])
        return ent

    # 计算用feat <= val划分数据集的信息增益
    def info_gain(self, X, Y, feat):
        # 划分前的熵
        N = len(Y)
        if N == 0:
            return 0
        HX = self.entropy(Y)
        unique_values = np.unique(X[:, feat])

        HXY = 0 # H(X|Y)
        for val in unique_values:
            # 分别计算every val of H(X|Y)
            Y_p = Y[X[:, feat] == val]
            HXY += len(Y_p) / len(Y) * self.entropy(Y_p)

        #return (HX - HXY)*100
        return (HX - HXY)

    def exponential_mechanism(self, T, attributes, epsilon, sensitivity=1):
        """
        实现指数机制，根据质量得分选择属性
        :param attributes: 候选属性列表
        :param epsilon: 隐私预算
        :param sensitivity: 质量函数的敏感度
        """
        X = T[:, :-1]
        Y = T[:, -1]
        scores = [self.info_gain(X, Y, attr_index) for attr_index in range(len(attributes))]
    
        # 处理可能的负得分（根据质量函数调整）
        min_score = min(scores)
        shifted_scores = [s - min_score for s in scores]  # 确保所有分数非负
    
        # 计算指数值
        exponents = [np.exp(epsilon * score / (2 * sensitivity)) for score in shifted_scores]
    
        probabilities = exponents / np.linalg.norm(exponents, ord=1)
        chosen_index = np.random.choice(len(attributes), p=probabilities)

        return chosen_index

    def select_random_features(self, feat_names, feat_status):
        """
        步骤8实现：从属性集中随机选择f个属性
        """
        #effective_random = int(np.sqrt(len(feat_names)))
        #return random.sample(feat_names, effective_random)

        # 筛选出所有离散型属性
        categorical_feats = [feat for feat in feat_names if feat_status.get(feat) == 'categorical']
    
        # 计算选择数量（离散属性数量的平方根）
        if categorical_feats:
            effective_random = max(1, int(np.sqrt(len(categorical_feats))))
            return random.sample(categorical_feats, effective_random)
        else:
            # 如果没有离散属性，返回空列表
            return []

    def get_path_features(self, node):
        """提取从根节点到当前叶节点的路径特征索引"""
        path_features = []

        current = node
        while current.parent:  # 假设节点添加了parent属性
            if current.feat not in path_features:
                if current.feat != '':
                    path_features.append(current.feat)
            current = current.parent
        
        #feats =  [self.log_feat_index[f] for f in path_features]
        #return sorted(feats) 
        return path_features

    def get_train_feature(self, feat_names, used_features):
        del_feats = []
        for feat in used_features:
            if self.feat_status[feat] == 'continuous':
                del_feats.append(feat)
        
        filtered_features = [
            feat for feat in feat_names 
            if feat not in del_feats
        ]
    
        return filtered_features


    def _train_logistic_leaf(self, node, epsilon, samples, T, feat_names):
        from sklearn.linear_model import LogisticRegression
    
        # 提取数据和路径特征
        """
        X = self.log_T[:, :-1].astype(float)
        y = self.log_T[:, -1].astype(int)
        path_features = self.get_path_features(node)
        X_selected = X[:, path_features]
        node.feature_indices = path_features
        """
        
        used_features = self.get_path_features(node)
        need_features = self.get_train_feature(feat_names, used_features)

        X = T[:, :-1].astype(float)
        y = T[:, -1].astype(int)
        #path_features = sorted([self.log_feat_index[f] for f in feat_names]) 
        path_features = sorted([self.log_feat_index[f] for f in need_features]) 
        X_selected = X[:, ]
        node.feature_indices = path_features
    
        # 训练逻辑回归模型
        model = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=1500,
            #max_iter=1000,
            #class_weight='balanced', # 处理类别不平衡
            random_state=42
        )

        model.fit(X_selected, y)
    
        # 添加差分隐私保护
        # TODO need update
        self._add_parameter_noise(model, samples, epsilon)
    
        # 存储模型
        node.logistic_model = model
    
    def _add_parameter_noise(self, model, n_samples, total_epsilon):
        """优化后的参数加噪方案"""
        # 1. 隐私预算分配
        coef_budget = total_epsilon * 0.7  # 70%给权重
        intercept_budget = total_epsilon * 0.3  # 30%给截距
    
        # 2. 权重系数加噪
        d = len(model.coef_[0])  # 特征数量
        for i in range(d):
            sensitivity = 1.0 / n_samples
            scale = sensitivity / coef_budget
            noise = np.random.laplace(0, scale)
            model.coef_[0][i] += noise
    
        # 3. 截距项加噪
        sensitivity_intercept = 1.0 / n_samples
        scale_intercept = sensitivity_intercept / intercept_budget
        noise_intercept = np.random.laplace(0, scale_intercept)
        model.intercept_ += noise_intercept
    
        # 4. 计算实际隐私消耗
        actual_epsilon = coef_budget + intercept_budget
        return actual_epsilon

    def predict(self, x, feat_names):
        node = self.root
        while node.child:
            index = feat_names.index(node.feat)
            feature_value = x[index]
            if feature_value in node.split:
                child_index = node.split[feature_value]
                node = node.child[child_index]
            else:
                # 最近邻处理
                known_values = list(node.split.keys())
                # 计算特征值距离（数值型特征）
                distances = [abs(feature_value - v) for v in known_values]
                # 或使用海明距离（类别型特征）
                # distances = [0 if feature_value == v else 1 for v in known_values]
            
                # 选择距离最小的分支
                min_index = np.argmin(distances)
                nearest_value = known_values[min_index]
                child_index = node.split[nearest_value]
                node = node.child[child_index]

        # 叶子节点预测
        if node.is_logistic and node.logistic_model:
            # 提取路径特征值
            x_path = [x[i] for i in node.feature_indices]
            ret = node.logistic_model.predict([x_path])[0]
            return 'a', ret
        else:
            return 'b', node.label  # 多数投票结果


    # 用ID3算法递归分裂结点，构造决策树
    def Build_DiffPID3(self, node, T, d, e, feat_names):

        t = self.get_max_A(T[:, :-1])
        Nt =  max(T.shape[0] + self.get_Noisy(e), 0)

        # step 7 
        #if t == -1 or d == 0 or self.get_heuristic_parameters(Nt, t, len(np.unique(T[:, -1:])), e):
        #if t == -1 or d == 0:
        if t == -1 or d == 0:
            e = self.Leaf_B

            # line 9 in Algorithm Differential Private ID3
            new_split_Y = self.partition_C(T[:, -1:])

            # 自适应决策机制
            num_label = len(np.unique(T[:, -1]))

            #if num_label == 1:
            #    print('what0 Nt: ', Nt, ' t ', t, ' d ', d, 'T.shape[0]', T.shape[0], ' num_label: ', num_label)

            # 逻辑回归需要至少两个类别才能训练分类模型
            #if len(T) >= self.regression_threshold and num_label >= 2:
            if Nt >= self.regression_threshold and num_label >= 2:
                #print('what Nt', Nt, len(T), self.regression_threshold)
                #self._train_logistic_leaf(node, e, len(T), T, feat_names)  # 逻辑回归
                self._train_logistic_leaf(node, e, Nt, T, feat_names)  # 逻辑回归
                node.is_logistic = True
                self.Leaf += 1
                self.log_Leaf += 1
                node.label = -2 
                #print('what1 Nt: ', Nt, ' t ', t, ' d ', d, 'T.shape[0]', T.shape[0], ' threshold', self.regression_threshold)
                return
        
            new_class = -1
            new_class_count = 0
            # line 10, 11 in Algorithm Differential Private ID3
            for value, sub_arr in new_split_Y.items():
                new_count = max(len(sub_arr) + self.get_Noisy(e), 0)
                if new_count >= new_class_count:
                    new_class = value
                    new_class_count = new_count
            
            #print('what2 Nt: ', Nt, ' t ', t, ' d ', d, ' new_class ', new_class, 'T.shape[0]', T.shape[0])
            node.label = new_class
            node.is_logistic = False 
            self.Leaf += 1
            return
        
        # TODO we support step 10 later.

        # step 8
        random_candidate_feat_names = self.select_random_features(feat_names, self.feat_status)
        #print(f"random candidate feat: {random_candidate_feat_names}")

        # step 11 
        random_New_split_A = self.exponential_mechanism(T, random_candidate_feat_names, e)
        
        New_split_A = -1
        cursor = 0
        while cursor < len(feat_names):
            if random_candidate_feat_names[random_New_split_A] == feat_names[cursor]:
                New_split_A = cursor
                break
            cursor += 1

        # TODO 连续属性：强制二元分裂  ​离散属性：多元分裂 , learn by <<DRPF: A Differential Privacy Proection Random Forest>>
        # line 15 in Algorithm Differential Private ID3
        if New_split_A != -1:
            New_T_dict = self.partition_A(T, New_split_A)
            index = 0
            node.feat = feat_names[New_split_A]
            # 创建新列表副本，避免共享引用
            new_feat_names = list(feat_names)  # 或 new_feat_names = feat_names.copy()
            del new_feat_names[New_split_A]
            for value, sub_arr in New_T_dict.items():
                #print(f"当第 {target_col_index} 列取值为 {value} 时，切分得到的子数组形状为: {sub_arr.shape}")
                new_node = Node()
                new_node.parent = node
                self.Build_DiffPID3(new_node, np.delete(sub_arr, New_split_A, axis=1), d-1, e, new_feat_names)
                node.split[value] = index 
                node.child.append(new_node)
                index += 1
        else:
            print('something error')

    # 计算在样本X，标签Y上的准确率
    def accuracy(self, X, Y, feat_names):
        correct = 0
        for x, y in zip(X, Y):
            pred = self.predict(x, feat_names)
            if pred == y:
                correct += 1
        return correct / len(Y)

    def visualize_tree(self, filename='dp_decision_tree_id3'):
        dot = Digraph(comment='Decision Tree')
        self.add_nodes_edges(dot, self.root)
        dot.render(filename, format='png', cleanup=True, view=True)
        print(f"决策树已保存为 {filename}.png")

    def add_nodes_edges(self, dot, node, parent_name=None, edge_label=None):
        if node.label is not None:
            node_name = str(id(node))
            dot.node(node_name, label=str(node.label))
            if parent_name is not None:
                dot.edge(parent_name, node_name, label=edge_label)
        else:
            node_name = str(id(node))
            dot.node(node_name, label=node.feat)
            if parent_name is not None:
                dot.edge(parent_name, node_name, label=edge_label)
            for value, child_index in node.split.items():
                child = node.child[child_index]
                self.add_nodes_edges(dot, child, node_name, str(value))