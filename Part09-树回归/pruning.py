#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import regtrees as reg
import tree_first as tf

def isTree(obj):
    """
    函数说明:判断测试输入变量是否是一棵树
    Parameters:
        obj - 测试对象
    Returns:
        是否是一棵树
    """
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    """
    函数说明:对树进行塌陷处理(即返回树平均值)
    Parameters:
        tree - 树
    Returns:
        树的平均值
    """
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    函数说明:后剪枝
    Parameters:
        tree - 树
        test - 测试集
    Returns:
        树的平均值
    """
    # 如果测试集为空,则对树进行塌陷处理
    if np.shape(testData)[0] == 0: return getMean(tree)
    # 如果有左子树或者右子树,则切分数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = tf.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 处理左子树(剪枝)
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    # 处理右子树(剪枝)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # 如果当前结点的左右结点为叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = tf.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(
            np.power(rSet[:, -1] - tree['right'], 2))
        # 计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并的误差小于没有合并的误差,则合并
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    train_filename = 'ex2.txt'
    train_Data = reg.loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = reg.createTree(train_Mat)
    print(tree)
    test_filename = 'ex2test.txt'
    test_Data = reg.loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print(prune(tree, test_Mat))
