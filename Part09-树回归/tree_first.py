#-*- coding:utf-8 -*-
import numpy as np


def binSplitDataSet(dataSet, feature, value):
    """
    函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
    """
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1


if __name__ == '__main__':
    testMat = np.mat(np.eye(4))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print('原始集合:\n', testMat)
    print('mat0:\n', mat0)
    print('mat1:\n', mat1)
