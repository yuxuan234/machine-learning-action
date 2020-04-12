#!/usr/bin/python
# -*- coding: UTF-8 -*-
import bayes_first as bf
import training_two as tt
import numpy as np


"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
        vec2Classify - 待分类的词条数组
        p0Vec - 侮辱类的条件概率数组
        p1Vec -非侮辱类的条件概率数组
        pClass1 - 文档属于侮辱类的概率
Returns:
        0 - 属于非侮辱类
        1 - 属于侮辱类
Modify:
    2020-04-11
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
     无
Returns:
     无
Modify:
    2020-04-11
"""


def testingNB():
    listOPosts, listClasses = bf.loadDataSet()  # 创建实验样本
    myVocabList = bf.createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bf.setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = tt.trainNB0(
        np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(bf.setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(bf.setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


if __name__ == '__main__':
    testingNB()
