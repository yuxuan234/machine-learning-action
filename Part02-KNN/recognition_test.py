#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import recognition as rt
import kNN_Two as kt
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN


"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无

Modify:
    2020-04-11
"""


def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = rt.img2vector('trainingDigits/%s' % (fileNameStr))
    # 构建kNN分类器
    # neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    # neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest = rt.img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果  classify函数-kNN算法,分类器
        classifierResult = kt.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


"""
函数说明:main函数

Parameters:
    无
Returns:
    无

Modify:
    2020-04-11
"""
if __name__ == '__main__':
    handwritingClassTest()
