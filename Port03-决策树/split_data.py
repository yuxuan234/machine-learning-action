#!/usr/bin/python
# -*- coding: UTF-8 -*-
import trees as tr

"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    无

Modify:
    2020-04-11
"""


def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回划分后的数据集


if __name__ == '__main__':
    dataSet, features = tr.createDataSet()
    print(splitDataSet(dataSet, 0, 1))
    print(splitDataSet(dataSet, 0, 0))
