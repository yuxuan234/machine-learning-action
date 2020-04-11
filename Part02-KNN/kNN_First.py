# -*- coding: UTF-8 -*-
import numpy as np

"""
函数说明:创建数据集

Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
Modify:
    2020-04-10
"""


def createDataSet():
    # 四组二维特征
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 打印数据集
    print(group)
    print(labels)
