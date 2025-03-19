import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Node:
    def __init__(self):
        # 内部结点的feat表示用来分类的特征编号，其数字与数据中的顺序对应
        # 叶结点的feat表示该结点对应的分类结果
        self.feat = None
        # 分类值列表，表示按照其中的值向子结点分类
        self.split = None
        # 子结点列表，叶结点的child为空
        self.child = []

class DecisionTree:
    def __init__(self, X, Y, feat_ranges, feat_names, lbd):
        self.root = Node()
        self.X = X
        self.Y = Y
        self.feat_ranges = feat_ranges # 特征取值范围
        self.feat_names = feat_names 
        self.lbd = lbd # 正则化系数
        self.eps = 1e-8 # 防止数学错误log(0)和除以0
        self.T = 0 # 记录叶结点个数
        self.ID3(self.root, self.X, self.Y)

    # 工具函数，计算 a * log a
    def aloga(self, a):
        return a * np.log2(a + self.eps)

    # 计算某个子数据集的熵
    def entropy(self, Y):
        cnt = np.unique(Y, return_counts=True)[1] # 统计每个类别出现的次数
        N = len(Y)
        ent = -np.sum([self.aloga(Ni / N) for Ni in cnt])
        return ent

    # 计算用feat <= val划分数据集的信息增益
    def info_gain(self, X, Y, feat, val):
        # 划分前的熵
        N = len(Y)
        if N == 0:
            return 0
        HX = self.entropy(Y)
        HXY = 0 # H(X|Y)
        # 分别计算H(X|X_F<=val)和H(X|X_F>val)
        Y_l = Y[X[:, feat] <= val]
        HXY += len(Y_l) / len(Y) * self.entropy(Y_l)
        Y_r = Y[X[:, feat] > val]
        HXY += len(Y_r) / len(Y) * self.entropy(Y_r)
        #HXY = self.entropy_YX(X, Y, feat, val)
        return HX - HXY

    # 计算特征feat <= val本身的复杂度H_Y(X)
    def entropy_YX(self, X, Y, feat, val):
        HYX = 0
        N = len(Y)
        if N == 0:
            return 0
        Y_l = Y[X[:, feat] <= val]
        HYX += -self.aloga(len(Y_l) / N)
        Y_r = Y[X[:, feat] > val]
        HYX += -self.aloga(len(Y_r) / N)
        return HYX

    # 计算用feat <= val划分数据集的信息增益率
    def info_gain_ratio(self, X, Y, feat, val):
        IG = self.info_gain(X, Y, feat, val)
        HYX = self.entropy_YX(X, Y, feat, val)
        return IG / HYX

    # 用ID3算法递归分裂结点，构造决策树
    def ID3(self, node, X, Y):
        # 判断是否已经分类完成
        if len(np.unique(Y)) == 1:
            node.feat = Y[0]
            self.T += 1
            return
        
        # 寻找最优分类特征和分类点
        best_IGR = 0
        best_feat = None
        best_val = None
        for feat in range(len(self.feat_names)):
            for val in self.feat_ranges[self.feat_names[feat]]:
                IGR = self.info_gain_ratio(X, Y, feat, val)
                if IGR > best_IGR:
                    best_IGR = IGR
                    best_feat = feat
                    best_val = val
        
        # 计算用best_feat <= best_val分类带来的代价函数变化
        # 由于分裂叶结点只涉及该局部，我们只需要计算分裂前后该结点的代价函数
        # 当前代价
        cur_cost = len(Y) * self.entropy(Y) + self.lbd
        # 分裂后的代价，按best_feat的取值分类统计
        # 如果best_feat为None，说明最优的信息增益率为0，
        # 再分类也无法增加信息了，因此将new_cost设置为无穷大
        if best_feat is None:
            new_cost = np.inf
        else:
            new_cost = 0
            X_feat = X[:, best_feat]
            # 获取划分后的两部分，计算新的熵
            new_Y_l = Y[X_feat <= best_val]
            new_cost += len(new_Y_l) * self.entropy(new_Y_l)
            new_Y_r = Y[X_feat > best_val]
            new_cost += len(new_Y_r) * self.entropy(new_Y_r)
            # 分裂后会有两个叶结点
            new_cost += 2 * self.lbd

        if new_cost <= cur_cost:
            # 如果分裂后代价更小，那么执行分裂
            node.feat = best_feat
            node.split = best_val
            l_child = Node()
            l_X = X[X_feat <= best_val]
            l_Y = Y[X_feat <= best_val]
            self.ID3(l_child, l_X, l_Y)
            r_child = Node()
            r_X = X[X_feat > best_val]
            r_Y = Y[X_feat > best_val]
            self.ID3(r_child, r_X, r_Y)
            node.child = [l_child, r_child]
        else:
            # 否则将当前结点上最多的类别作为该结点的类别
            vals, cnt = np.unique(Y, return_counts=True)
            node.feat = vals[np.argmax(cnt)]
            self.T += 1

    # 预测新样本的分类
    def predict(self, x):
        node = self.root
        # 从根结点开始向下寻找，到叶结点结束
        while node.split is not None:
            # 判断x应该处于哪个子结点
            if x[node.feat] <= node.split:
                node = node.child[0]
            else:
                node = node.child[1]
        # 到达叶结点，返回类别
        return node.feat

    # 计算在样本X，标签Y上的准确率
    def accuracy(self, X, Y):
        correct = 0
        for x, y in zip(X, Y):
            pred = self.predict(x)
            if pred == y:
                correct += 1
        return correct / len(Y)