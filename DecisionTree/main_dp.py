from helper_dp import process
import numpy as np
from SuLQID3 import SulQID3

if __name__ == '__main__':

    train_x, train_y, test_x, test_y, feat_ranges, feat_names = process()

    print(feat_ranges)
    #print(feat_names.tolist())
    print(train_x.shape)
    print(train_y.shape)
    #print(test_x.shape)
    #print(test_y.shape)
    #for key, value in feat_ranges.items():
    #    print(key, len(value))


    DPDT = SulQID3(train_x, train_y, feat_names, 5, 1)
    #DPDT = SulQID3(train_x, train_y, feat_names, 5, 0.75)
    #DPDT = SulQID3(train_x, train_y, feat_names, 5, 0.25)
    #DPDT = SulQID3(train_x, train_y, feat_names, 5, 3.25)
    #DPDT = SulQID3(train_x, train_y, feat_names, 5, 0.5)
    #DPDT = SulQID3(train_x, train_y, feat_names, 5, 5.25)
    print('叶结点数量：', DPDT.Leaf)

    # 计算在训练集和测试集上的准确率
    print('训练集准确率：', DPDT.accuracy(train_x, train_y, feat_names.tolist()))
    print('测试集准确率：', DPDT.accuracy(test_x, test_y, feat_names.tolist()))

    DPDT.visualize_tree()

    print('done')