from helper_dp import process
import numpy as np
from RandomForest import RandomForest

if __name__ == '__main__':

    train_x, train_y, test_x, test_y, feat_ranges, feat_names = process()

    print(feat_ranges)
    print(feat_names)
    #print(feat_names.tolist())
    #print(type(feat_names))
    #for key, value in feat_ranges.items():
    #    print(key, len(value))


    #RF = RandomForest(train_x, train_y, feat_names, 5, 3.25, 1)
    RF = RandomForest(train_x, train_y, feat_names, 5, 3.25, 10)
    #DPDT = DiffPID3(train_x, train_y, feat_names, 5, 0.75)
    #DPDT = DiffPID3(train_x, train_y, feat_names, 5, 0.25)
    #DPDT = DiffPID3(train_x, train_y, feat_names, 5, 0.5)
    #RF = RandomForest(train_x, train_y, feat_names, 5, 3.25, 10)
    #DPDT = DiffPID3(train_x, train_y, feat_names, 5, 7.25)
    #DPDT = DiffPID3(train_x, train_y, feat_names, 5, 5.25)
    #print('叶结点数量：', DPDT.Leaf)

    # 计算在训练集和测试集上的准确率
    print('训练集准确率：', RF.accuracy(train_x, train_y, feat_names.tolist()))
    print('测试集准确率：', RF.accuracy(test_x, test_y, feat_names.tolist()))

    #DPDT.visualize_tree()

    print('done')