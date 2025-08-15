import pandas as pd
import numpy as np
from collections import Counter
import threading
import queue
from DiffPRFM import DiffPRFM
from DiffPM import DiffPM
from sklearn.utils import resample

# This class represents an ensemble structe of private random decision trees.
# It allows the parallel creation and training of the contained trees.
#
# Parameters for constructor:
# number_trees: Number of random decision trees, which should be used
# max_depth: The depth of each decision tree
# attributes: The full list of available attributes from which the decision
#     trees can choose from
# e: the epsilon parameter for the Laplacian distribution
class RandomForest(object):
    def __init__(self, X, Y, feat_names, feat_status, max_depth, e, number_trees):
        self.trees = list() #list of containing trees
        # the more trees are used, the higher the noise
        self.e_per_tree = e / number_trees

        # 第3步实现：Bootstrap采样
        bootstrap_samples = []
        for i in range(number_trees):
            # 有放回随机采样，保持原始数据集大小
            X_i, Y_i = resample(X, Y, replace=True, n_samples=len(X))
            bootstrap_samples.append((X_i, Y_i))

        # Queue for the output of the parallel threads
        que = queue.Queue() 
        for i in range(0, number_trees):
            X_i, Y_i = bootstrap_samples[i]
            thr = threading.Thread(target = lambda q, 
                    arg : q.put(DiffPM(arg)), 
                    #arg : q.put(DiffPRFM(arg)), 
                    args = (que, [X_i, Y_i, feat_names, feat_status, max_depth, self.e_per_tree, i]))
            thr.start()
            thr.join()
        
        while len(self.trees) < number_trees: 
            # wait until all trees are build and trained 
            self.trees.append(que.get())
    
    def predict(self, x, feat_names):
        """
        对输入样本进行随机森林预测
        参数:
            x: one sample
        返回:
            predictions: 预测结果数组 (n_samples,)
        """
        # 收集所有树的预测结果
        tree_predictions = []
        modle_type = 'c'
        most_type = []
        for tree in self.trees:
            # 调用每棵树的预测方法（需确保DiffPID3.predict支持单样本输入）
            modle_type, pred = tree.predict(x, feat_names)
            tree_predictions.append(pred)
            most_type.append(modle_type)
        
        # 多数投票决策
        pred_counter = Counter(tree_predictions)
        most_counter = Counter(most_type)
        return pred_counter.most_common(1)[0][0], most_counter.most_common(1)[0][0] 
    
    def accuracy(self, X, Y, feat_names):
        """
        计算随机森林在测试集上的准确率
        参数:
            X: 测试集特征矩阵 (n_samples, n_features)
            Y: 测试集真实标签 (n_samples,)
        返回:
            accuracy: 模型准确率 (0-1之间)
        """

        correct = 0
        total = len(Y)
        ret_None = 0

        error_log = 0
        error_total = 0
        error_norm = 0
        right_log = 0
        right_norm = 0
        
        # 遍历每个样本
        for i in range(total):
            # 获取单个样本
            x = X[i]
            true_label = Y[i]
            
            # 预测单个样本
            pred, log_type = self.predict(x, feat_names)

            if pred == None:
                ret_None += 1
            
            # 检查预测是否正确
            if pred == true_label:
                correct += 1

                if log_type == 'a':
                    right_log += 1

                if log_type == 'b':
                    right_norm += 1
            
            if pred != true_label:
                error_total += 1 

                if log_type == 'a':
                    error_log += 1
                
                if log_type == 'b':
                    error_norm += 1

        print('Return None: ', ret_None)
        print('correct: ', correct)
        print('Total: ', total)
        print('error_total: ', error_total)
        print('error_log: ', error_log)
        print('right_log: ', right_log)
        print('error_norm: ', error_norm)
        print('right_norm: ', right_norm)
        if right_log + error_log > 0:
            print('ratio log: ', right_log / (right_log + error_log))
        if right_norm + error_norm > 0:
            print('ratio norm: ', right_norm / (right_norm + error_norm))

        return correct / total

    def visualize_tree(self):
        for tree in self.trees:
             tree.visualize_tree()