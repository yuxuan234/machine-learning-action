#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import annotation_tree as at

"""
函数说明:存储决策树

Parameters:
    inputTree - 已经生成的决策树
    filename - 决策树的存储文件名
Returns:
    无
Modify:
    2020-04-11
"""


def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    myTree = at.retriveTree(0)
    storeTree(myTree, 'classifierStorage.txt')
    myTree = grabTree('classifierStorage.txt')
    print(myTree)
