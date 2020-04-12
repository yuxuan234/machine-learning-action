#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import random
import draw_two as dt
import logregres as lg

"""
函数说明:改进的随机梯度上升算法

Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
Modify:
    2020-04-12
"""


def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01  # 降低alpha的大小，每次减小1/(j+i)。
    weights = np.ones(n)  # 参数初始化
    for i in range(m):
        h = lg.sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h  # 计算误差
        weights = weights + alpha * error * dataMatrix[i]  # 更新回归系数
    return weights  # 返回


"""
函数说明:改进的随机梯度上升算法

Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
Modify:
    2020-04-12
"""
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)  # 参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            # 选择随机选取的一个样本，计算h
            h = lg.sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            del(dataIndex[randIndex])  # 删除已经使用的样本
    return weights


if __name__ == '__main__':
    dataMat, labelMat = lg.loadDataSet()
    # weights = stocGradAscent0(np.array(dataMat), labelMat)
    # dt.plotBestFit(weights)

    weights = stocGradAscent1(np.array(dataMat), labelMat)
    dt.plotBestFit(weights)
