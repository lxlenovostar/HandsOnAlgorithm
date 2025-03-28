import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from graphviz import Digraph
from helper import SplitCriteria


class Node:
    def __init__(self):
        # 内部结点的feat表示用来分类的特征编号，其数字与数据中的顺序对应
        self.feat = None
        # 分类值列表，表示按照其中的值向子结点分类, key is value, value is index of self.child
        self.split = {} 
        # 子结点列表，叶结点的child为空
        self.child = []
        # 叶结点的label表示该结点对应的分类结果
        self.lable = None

# From paper of Arik Friedman <<Data Mining with Differential Privacy>>
class SulQID3:
    def __init__(self, X, Y, feat_ranges, feat_names, d, B):
        self.root = Node()
        self.T = np.hstack((X, Y.reshape(-1, 1)))
        self.feat_ranges = feat_ranges # 特征取值范围
        self.feat_names = feat_names 
        self.d = d # d + 1 the depth of tree
        self.B = B # differential privacy budget
        self.e = self.B / (d + 1)
        self.eps = 1e-8 # 防止数学错误log(0)和除以0
        self.Leaf = 0 # 记录叶结点个数

        self.Build_Sulq_ID3(self.root, self.T, self.d, self.e, self.feat_names.tolist())

    def get_max_A(self, X):
        # 遍历每一列
        max_A_len = 0
        index_A  = -1
        for i in range(X.shape[1]):
            # 获取当前列
            column = X[:, i]
            # 找出该列的唯一值
            unique_values_len = len(np.unique(column))
            if max_A_len < unique_values_len:
                max_A_len = unique_values_len
                index_A = i
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
        # line 9 in Algorithm SuLQ-based ID3
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
    
    def partiton_A_C(self, T):
        # 存储切分结果的字典
        split_results = {}

        # subtract column col of label 
        for col_idx in range(T.shape[1] - 1):
            # 获取该列的唯一取值
            unique_values = np.unique(T[:, col_idx])
            # 获取标签列的唯一取值
            unique_labels = np.unique(T[:, -1])
            # 为该列创建一个子字典来存储切分结果
            split_results[col_idx] = {}
            for value in unique_values:
                split_results[col_idx][value] = {}
                for label in unique_labels:
                    # 根据该列的取值和标签进行切分
                    mask_attr = T[:, col_idx] == value
                    mask_label = T[:, -1] == label
                    mask = mask_attr & mask_label
                    split_results[col_idx][value][label] = T[mask]

        # 打印切分结果的示例
        """
        for col_idx in split_results:
            for value in split_results[col_idx]:
                for label in split_results[col_idx][value]:
                    print(f"第 {col_idx} 列，取值为 {value}，标签为 {label} 的子数组形状: {split_results[col_idx][value][label].shape}")
        """
        return split_results

    # 用ID3算法递归分裂结点，构造决策树
    def Build_Sulq_ID3(self, node, T, d, e, feat_names):

        t = self.get_max_A(T[:, :-1])
        Nt =  T.shape[0] + self.get_Noisy(e)

        print('debug depth:', d, ' t:', t)
        if t == 0 or d == 0 or self.get_heuristic_parameters(Nt, t, len(np.unique(T[:, -1:])), e):
            # line 10 in Algorithm SuLQ-based ID3
            print('debug depth end1:', d, ' t:', t)
            new_split_Y = self.partition_C(T[:, -1:])

            new_class = -1
            new_class_count = 0
            # line 11, 12 in Algorithm SuLQ-based ID3
            for value, sub_arr in new_split_Y.items():
                new_count = len(sub_arr) + self.get_Noisy(e)
                if new_count > new_class_count:
                    new_class = value
                    new_class_count = new_count
            
            node.label = new_class
            self.Leaf += 1
            return
        
        print('debug depth end2:', d, ' t:', t)
        # line 14, 15, 16 in Algorithm SuLQ-based ID3
        new_split_A_Y = self.partiton_A_C(T)
        NUM_A = T.shape[1] - 1
        Value_A = -np.inf 
        New_split_A = -1 
        for col_idx in new_split_A_Y:
            current_value_A = 0 
            for value in new_split_A_Y[col_idx]:
                N_j = 0
                N_j_c_Noise = []
                for label in new_split_A_Y[col_idx][value]:
                    N_j_c = len(new_split_A_Y[col_idx][value][label])
                    N_j += N_j_c
                    # line 18 in Algorithm SuLQ-based ID3
                    N_j_c_Noise.append(N_j_c + self.get_Noisy(e / (2*NUM_A)))
                    #print(f"第 {col_idx} 列，取值为 {value}，标签为 {label} 的子数组形状: {new_split_A_Y[col_idx][value][label].shape}")
                # line 17 in Algorithm SuLQ-based ID3
                N_j += self.get_Noisy(e / (2*NUM_A))
                for n_j_c in N_j_c_Noise:
                    if n_j_c <= 0 or N_j <= 0:
                        continue
                    # line 19 in Algorithm SuLQ-based ID3
                    current_value_A += n_j_c *  math.log2(n_j_c / N_j) 
            if current_value_A >= Value_A:
                Value_A = current_value_A
                New_split_A = col_idx

        # line 22 in Algorithm SuLQ-based ID3
        if New_split_A != -1:
            New_T_dict = self.partition_A(T, New_split_A)
            index = 0
            print(T.shape)
            print(type(feat_names))
            print(feat_names)
            print('debug New_split_A:', New_split_A)
            node.feat = feat_names[New_split_A]
            # 创建新列表副本，避免共享引用
            new_feat_names = list(feat_names)  # 或 new_feat_names = feat_names.copy()
            del new_feat_names[New_split_A]
            print(type(new_feat_names))
            for value, sub_arr in New_T_dict.items():
                #print(f"当第 {target_col_index} 列取值为 {value} 时，切分得到的子数组形状为: {sub_arr.shape}")
                new_node = Node()
                print('debug new_feat_names check', new_feat_names, sub_arr.shape)
                self.Build_Sulq_ID3(new_node, np.delete(sub_arr, New_split_A, axis=1), d-1, e, new_feat_names)
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
                # 如果特征值不在分割条件中，可以根据需求进行处理，这里简单返回 None
                return None
        return node.label

    # 计算在样本X，标签Y上的准确率
    def accuracy(self, X, Y, feat_names):
        correct = 0
        for x, y in zip(X, Y):
            pred = self.predict(x, feat_names)
            if pred == y:
                correct += 1
        return correct / len(Y)

    def visualize_tree(self, filename="decision_tree"):
        """将决策树可视化并保存为PNG文件"""
        dot = Digraph(comment='Decision Tree')
        self._add_nodes(dot, self.root)
        dot.render(filename, format='png', cleanup=True)
        print(f"决策树已保存为 {filename}.png")

    def _add_nodes(self, dot, node, parent=None, edge_label=""):
        """递归添加节点到graphviz图"""
        node_id = str(id(node))  # 使用对象id作为唯一标识
        
        if node.lable is not None:  # 叶节点
            dot.node(node_id, f'Class: {node.lable}')
        else:  # 内部节点
            feature_name = self.feat_names[node.feat]
            dot.node(node_id, f'{feature_name} Split Value')
        
        if parent is not None:
            dot.edge(parent, node_id, label=edge_label)
        
        # 递归添加子节点
        if not node.child:  # 叶节点不需要子节点
            return
        
        for value, child_idx in node.split.items():
            child_node = node.child[child_idx]
            self._add_nodes(dot, child_node, parent=node_id, 
                          edge_label=f'Value: {value}')
