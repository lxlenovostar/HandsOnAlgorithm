import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 
import random
from graphviz import Digraph
from typing import List, Union
from scipy.stats import truncnorm


class Node:
    def __init__(self):
        # 内部结点的feat表示用来分类的特征编号，其数字与数据中的顺序对应
        self.feat = '' 
        # 分类值列表，表示按照其中的值向子结点分类, key is value, value is index of self.child
        self.split = {} 
        # 子结点列表，叶结点的child为空
        self.child = []
        # 叶结点的label表示该结点对应的分类结果
        self.label = None
        # 表示此节点分裂时是不是用了连续属性
        self.status_cont = False
        # 连续属性的分裂点要单独存储，因为一个子树是小于等于，一个子树是大于。
        self.cont_split_point = None 

class DiffPM:
    # line 2 in Algorithm 2 Differential private ID3
    # Y for C, feat_names for A
    def __init__(self, inputs):

        X = inputs[0]
        Y = inputs[1]
        feat_names = inputs[2]
        feat_status = inputs[3]
        d = inputs[4]
        B = inputs[5]
        tree_id = inputs[6]

        self.root = Node()
        self.T = self.get_sample_dataset(X, Y, tree_id) 
        self.feat_names = feat_names 
        self.feat_status = feat_status 
        self.d = d # d + 1 the depth of tree
        self.B = B # differential privacy budget
        self.e = self.B / (2*(d + 1))
        self.Leaf = 0 # 记录叶结点个数

        entropy_seed = int(time.time() * 1000) + tree_id * 10000
        random.seed(entropy_seed)
        
        self.Build_DiffPID3(self.root, self.T, self.d, self.e, self.feat_names.tolist())

        print('number of tree leaf: ', self.Leaf)
    
    def get_sample_dataset(self, X, Y, tree_id):
        # 新增随机采样逻辑
        n_samples = X.shape[0]
        rng = np.random.RandomState(int(time.time()) + tree_id)
        indices = rng.choice(n_samples, size=n_samples, replace=True)  # 有放回抽样
        X_sampled = X[indices]
        Y_sampled = Y[indices]

        return np.hstack((X_sampled, Y_sampled.reshape(-1, 1)))

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
        truncation = 3.0
        scale = 1 / e
        a, b = -truncation, truncation
        tn = truncnorm((a - 0)/scale, (b - 0)/scale, loc=0, scale=scale)
        return tn.rvs()
    
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
    
    def partition_A(self, T, index, status, split_point=None):
        """
        增强版分区函数：支持离散和连续属性
    
        参数:
        T: 数据集 (numpy数组)
        index: 目标特征列索引
        status: 属性类型 (True:连续, False:离散)
        split_point: 连续属性的分割点 (仅当status=True时有效)
    
        返回:
        分区结果字典
        """
        # 参数验证
        if index >= T.shape[1]:
            raise ValueError(f"索引{index}超出数据维度{T.shape[1]}")
    
        if status and split_point is None:
            raise ValueError("连续属性必须提供split_point")
    
        # 离散属性处理
        if not status:
            # 找到数组中的唯一值
            unique_values = np.unique(T[:, index])
        
            # 用于存储切分结果的字典
            split_arrays = {}
        
            # 遍历每个唯一取值
            for value in unique_values:
                # 创建布尔掩码
                mask = T[:, index] == value
                # 根据掩码选取子集
                split_arrays[value] = T[mask]
        
            return split_arrays
    
        # 连续属性处理 (DiffPRFs核心创新)
        else:
            # 创建左右子集掩码
            left_mask = T[:, index].astype(float) <= split_point
            right_mask = ~left_mask
        
            # 构建分区结果
            split_arrays = {
                'left': T[left_mask],
                'right': T[right_mask]
            }
        
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

        return (HX - HXY)

    def exponential_mechanism(self, T, attributes, epsilon, continuos_split_point, sensitivity=1):
        """
        实现指数机制，根据质量得分选择属性
        :param attributes: 候选属性列表
        :param epsilon: 隐私预算
        :param sensitivity: 质量函数的敏感度
        """
        X = T[:, :-1]
        Y = T[:, -1]

        scores = []
        for attr_index in range(len(attributes)):
            if attributes[attr_index] in continuos_split_point:
                feature_values = X[:, attr_index].astype(float)
                scores.append(self.continuous_info_gain(feature_values, Y, continuos_split_point[attributes[attr_index]])) 
            else:
                scores.append(self.info_gain(X, Y, attr_index)) 

        # 处理可能的负得分（根据质量函数调整）
        min_score = min(scores)
        shifted_scores = [s - min_score for s in scores]  # 确保所有分数非负
    
        # 计算指数值
        exponents = [np.exp(epsilon * score / (2 * sensitivity)) for score in shifted_scores]
    
        probabilities = exponents / np.linalg.norm(exponents, ord=1)
        chosen_index = np.random.choice(len(attributes), p=probabilities)

        return chosen_index

    def select_random_features(self, feat_names):
        """
        步骤8实现：从属性集中随机选择f个属性
        """
        effective_random = int(np.sqrt(len(feat_names)))
        
        return random.sample(feat_names, effective_random)
    
    def handle_continuous_feat(self, T, feat_names, continuous_feat, epsilon):
        cont_split = {}
        X = T[:, :-1]
        Y = T[:, -1]
        for index in range(len(feat_names)):
            feat = feat_names[index]
            if feat in continuous_feat:
                cont_split[feat] = self.step10_continuous_split(X, Y, index, epsilon)

        return cont_split
    
    def check_continuous_feat(self, candidate_feat_names):
        ret = False
        ret_cont = []

        for feat in candidate_feat_names:
            if self.feat_status[feat] == 'continuous':
                ret = True
                ret_cont.append(feat)

        return ret, ret_cont 

    def step10_continuous_split(self, X, Y, feature_idx, epsilon, delta_q=1.0, n_bins=10):
        """
        实现DiffPRFs算法第10步：连续属性分裂点选择（增强版）
    
        参数:
        X: 特征矩阵 (n_samples, n_features)
        Y: 目标向量 (n_samples,)
        feature_idx: 当前处理的连续属性索引
        epsilon: 当前节点分配的隐私预算
        delta_q: 打分函数敏感度 (默认1.0)
        n_bins: 候选区间数量 (默认10)
    
        返回:
        best_split: 选择的最佳分裂点
        """
        
        # 提取当前连续属性值
        feature_values = X[:, feature_idx].astype(float)
    
        # 1. 生成候选分裂区间 (等频分箱)
        sorted_values = np.sort(feature_values)
        bin_edges = np.percentile(sorted_values, np.linspace(0, 100, n_bins+1))
    
        # 2. 计算每个区间的打分和长度
        q_scores = []      # 打分函数值
        interval_sizes = [] # 区间长度 |R_i|
        split_points = []   # 候选分裂点
        all_zeros = True   # 新增：全零标志

        for i in range(len(bin_edges)-1):
            low, high = bin_edges[i], bin_edges[i+1]
            split_point = (low + high) / 2  # 取区间中点为候选分裂点
            split_points.append(split_point)
    
            # 计算区间长度
            interval_size = high - low
            interval_sizes.append(interval_size)
    
            # 计算打分函数 (信息增益)
            mask = (feature_values >= low) & (feature_values <= high)
            y_subset = Y[mask]
    
            if len(y_subset) == 0:
                q_score = 0
            else:
                # 使用信息增益作为打分函数
                q_score = self.continuous_info_gain(feature_values, Y, split_point)
                if q_score != 0:
                    all_zeros = False  # 发现非零值
        
            q_scores.append(q_score)
    
        # 3. 处理全零信息增益场景
        if all_zeros:
            # 策略1：随机选择分裂点（保持隐私保护）
            #best_split = np.random.choice(split_points)
        
            # 策略2：返回特征中位数（更稳定）
            best_split = np.median(feature_values)
        
            return best_split
    
        # 4. 正常流程：计算加权概率
        exp_values = np.exp(epsilon/(2*delta_q) * np.array(q_scores)) * np.array(interval_sizes)
        probabilities = exp_values / np.sum(exp_values)
    
        # 5. 按概率选择分裂点
        selected_idx = np.random.choice(len(split_points), p=probabilities)
        best_split = split_points[selected_idx]

        return best_split

    def continuous_info_gain(self, feature_values, y, split_point):
        """
        计算信息增益 (打分函数)
        """
        # 划分数据集
        left_mask = feature_values <= split_point
        right_mask = ~left_mask
    
        # 计算父节点熵
        parent_entropy = self.entropy(y)
    
        # 计算子节点熵
        n_total = len(y)
        n_left = np.sum(left_mask)
        n_right = n_total - n_left
    
        if n_left == 0 or n_right == 0:
            return 0
    
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
    
        # 计算条件熵
        conditional_entropy = (n_left/n_total)*left_entropy + (n_right/n_total)*right_entropy
    
        # 信息增益
        return parent_entropy - conditional_entropy


    # 用ID3算法递归分裂结点，构造决策树
    def Build_DiffPID3(self, node, T, d, e, feat_names):

        t = self.get_max_A(T[:, :-1])
        Nt =  max(T.shape[0] + self.get_Noisy(e), 0)
        #Nt = max(len(np.unique(T[:, -1:])) + self.get_Noisy(e), 0)
        # TODO step 6
        # step 7 
        #if t == -1 or d == 0 or self.get_heuristic_parameters(Nt, t, len(np.unique(T[:, -1:])), e):
        #if t == -1 or d == 0 :
        # TODO need update
        #if len(feat_names) == 0 or d == 0 or self.get_heuristic_parameters(Nt, t, len(np.unique(T[:, -1:])), e):

        #if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1:
        #min_samples_leaf = max(self.limit_sample, int(0.01 * len(T)))
        #if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1 or Nt < (min_samples_leaf * 2):
        #if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1:
        #if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1 or self.get_heuristic_parameters(Nt, t, len(np.unique(T[:, -1:])), e):
        #min_samples_leaf = max(10, int(0.01 * len(T)))  # 至少5个样本
        #min_samples_leaf = min(50, int(0.02 * len(T)))  # 至少5个样本
        #if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1 or Nt <= min_samples_leaf:
        #if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1:
        #if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1 or Nt <= 50:
        if len(feat_names) == 0 or d == 0 or len(np.unique(T[:, -1:])) == 1:
            #print('what Nt ', Nt, ' T.shape[0] ', T.shape[0], ' e ', e, ' noise ', self.get_Noisy(e))
            # line 9 in Algorithm Differential Private ID3
            new_split_Y = self.partition_C(T[:, -1:])

            new_class = -1
            new_class_count = 0
            noise  = self.get_Noisy(e)
            # line 10, 11 in Algorithm Differential Private ID3
            for value, sub_arr in new_split_Y.items():
                new_count = max(len(sub_arr) + noise, 0)
                if new_count >= new_class_count:
                    new_class = value
                    new_class_count = new_count
            
            node.label = new_class
            self.Leaf += 1
            return
        
        # step 8
        random_candidate_feat_names = self.select_random_features(feat_names)
        #random_candidate_feat_names = feat_names
        #random_candidate_feat_names = ['hours_per_week'] 
        #print(f"random candidate feat: {random_candidate_feat_names}")
        
        # step 9 
        continuous_status, continuous_feat = self.check_continuous_feat(random_candidate_feat_names)
        # step 10
        continuos_split_point = {}
        if continuous_status:
             cont_len = len(continuous_feat)
             e = e / (cont_len + 1)
             continuos_split_point = self.handle_continuous_feat(T, feat_names, continuous_feat, e)

        # step 11 
        random_New_split_A = self.exponential_mechanism(T, random_candidate_feat_names, e, continuos_split_point)
        
        status_cont =  False
        New_split_A = -1
        cursor = 0
        while cursor < len(feat_names):
            if random_candidate_feat_names[random_New_split_A] == feat_names[cursor]:
                New_split_A = cursor
                break
            cursor += 1
        
        if random_candidate_feat_names[random_New_split_A] in continuos_split_point:
            status_cont =  True 

        if New_split_A != -1:
            New_T_dict = {}
            if status_cont: 
                New_T_dict = self.partition_A(T, New_split_A, status_cont, continuos_split_point[feat_names[New_split_A]])
                node.status_cont = True
                node.cont_split_point = continuos_split_point[feat_names[New_split_A]]
            else:
                New_T_dict = self.partition_A(T, New_split_A, status_cont, None)

            index = 0
            node.feat = feat_names[New_split_A]
            # 创建新列表副本，避免共享引用
            new_feat_names = list(feat_names)  # 或 new_feat_names = feat_names.copy()

            if status_cont == False:
                del new_feat_names[New_split_A]

            for value, sub_arr in New_T_dict.items():
                #print(f"当第 {target_col_index} 列取值为 {value} 时，切分得到的子数组形状为: {sub_arr.shape}")
                if sub_arr.shape[0] == 0:
                    continue
                new_node = Node()
                # 连续属性：强制二元分裂,保留连续属性  ​离散属性：多元分裂
                if status_cont:
                    self.Build_DiffPID3(new_node, sub_arr, d-1, self.e, new_feat_names)
                else:
                    self.Build_DiffPID3(new_node, np.delete(sub_arr, New_split_A, axis=1), d-1, self.e, new_feat_names)
                node.split[value] = index 
                node.child.append(new_node)
                index += 1
        else:
            print('something error')

    def predict(self, x, feat_names):
        """
        支持连续和离散属性
    
        参数:
        x: 输入特征向量
        feat_names: 特征名称列表
    
        返回:
        预测结果元组 (模型标识, 预测标签)
        """
        node = self.root
    
        # 遍历决策树直到叶节点
        while node.child:
            # 获取特征索引
            index = feat_names.index(node.feat)
            feature_value = x[index]
        
            # 连续属性处理
            if node.status_cont:  # 连续属性标志
                # 与分裂点比较
                child_index = -1 
                if len(node.split) == 1:
                    child_index = node.split['left']  # 左分支
                else:
                    if feature_value <= node.cont_split_point:
                        child_index = node.split['left']  # 左分支
                    else:
                        child_index = node.split['right']  # 右分支
                node = node.child[child_index]
        
            # 离散属性处理
            else:
                # 直接匹配
                if feature_value in node.split:
                    child_index = node.split[feature_value]
                    node = node.child[child_index]
                # 未知值处理（最近邻）
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
     
        return 'DiffPM', node.label

    # 计算在样本X，标签Y上的准确率
    def accuracy(self, X, Y, feat_names):
        correct = 0
        for x, y in zip(X, Y):
            pred = self.predict(x, feat_names)
            if pred == y:
                correct += 1
        return correct / len(Y)

    def visualize_tree(self, filename='DiffPMs'):
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