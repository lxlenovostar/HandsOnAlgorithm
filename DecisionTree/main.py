from C45 import DecisionTree
from helper import process

if __name__ == '__main__':

    train_x, train_y, test_x, test_y, feat_ranges, feat_names = process()

    DT = DecisionTree(train_x, train_y, feat_ranges, feat_names, lbd=1.0)
    print('叶结点数量：', DT.T)

    # 计算在训练集和测试集上的准确率
    print('训练集准确率：', DT.accuracy(train_x, train_y))
    print('测试集准确率：', DT.accuracy(test_x, test_y))

    # 可视化决策树
    DT.visualize()
    print('done')