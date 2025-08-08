import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 
import random
from graphviz import Digraph
from typing import List, Union


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

class DiffPM:
    # line 2 in Algorithm 2 Differential private ID3
    # Y for C, feat_names for A
    def __init__(self, inputs):

        X = inputs[0]
        Y = inputs[1]
        feat_names = inputs[2]
        d = inputs[3]
        B = inputs[4]
        tree_id = inputs[5]

        self.root = Node()
        self.T = self.get_sample_dataset(X, Y, tree_id) 
        self.feat_names = feat_names 
        self.d = d # d + 1 the depth of tree
        self.B = B # differential privacy budget
        self.e = self.B / (2*(d + 1))
        self.Leaf = 0 # 记录叶结点个数
        
        np.random.seed(int(time.time()) + tree_id)
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
        sensitivity = 1
        epsilon = e
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)

        return noise
    
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

        return (HX - HXY)*100

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

    def select_random_features(self, feat_names):
        """
        步骤8实现：从属性集中随机选择f个属性
        """
        effective_random = int(np.sqrt(len(feat_names)))
        
        return random.sample(feat_names, effective_random)

    # 用ID3算法递归分裂结点，构造决策树
    def Build_DiffPID3(self, node, T, d, e, feat_names):

        t = self.get_max_A(T[:, :-1])
        Nt =  max(T.shape[0] + self.get_Noisy(e), 0)

        # step 7 
        #if t == -1 or d == 0 or self.get_heuristic_parameters(Nt, t, len(np.unique(T[:, -1:])), e):
        if t == -1 or d == 0 :
            # line 9 in Algorithm Differential Private ID3
            new_split_Y = self.partition_C(T[:, -1:])

            new_class = -1
            new_class_count = 0
            # line 10, 11 in Algorithm Differential Private ID3
            for value, sub_arr in new_split_Y.items():
                new_count = max(len(sub_arr) + self.get_Noisy(e), 0)
                if new_count >= new_class_count:
                    new_class = value
                    new_class_count = new_count
            
            node.label = new_class
            self.Leaf += 1
            return
        
        # TODO we support step 10 later.

        # step 8
        random_candidate_feat_names = self.select_random_features(feat_names)
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
                self.Build_DiffPID3(new_node, np.delete(sub_arr, New_split_A, axis=1), d-1, e, new_feat_names)
                node.split[value] = index 
                node.child.append(new_node)
                index += 1
        else:
            print('something error')

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
        return 'DiffPM', node.label

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