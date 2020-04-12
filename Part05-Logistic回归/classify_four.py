#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import logregres as lg
import improve_three as it


"""
函数说明:分类函数

Parameters:
    inX - 特征向量
    weights - 回归系数
Returns:
    分类结果
Modify:
    2020-04-12
"""


def classifyVector(inX, weights):
    prob = lg.sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


"""
函数说明:使用Python写的Logistic分类器做预测

Parameters:
    无
Returns:
    无
Modify:
    2020-04-12
"""


def colicTest():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        # print(len(currLine))
        lineArr = []
        for i in range(len(currLine) - 1):
            # print(float(currLine[i]))
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        # print(float(currLine[-1]) == float(currLine[21]))
        trainingLabels.append(float(currLine[-1]))
    trainWeights = it.stocGradAscent1(
        np.array(trainingSet),
        trainingLabels, 500)  # 使用改进的随即上升梯度训练
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100  # 错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("迭代%d后的平均错误率为: %.2f%%" % (numTests, float(errorSum/float(numTests))))


if __name__ == '__main__':
    multiTest()
