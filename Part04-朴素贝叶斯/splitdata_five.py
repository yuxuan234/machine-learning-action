#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import numpy as np
import random
import bayes_first as bf
import optimize_four as of
import testing_three as te

"""
函数说明:接收一个大字符串并将其解析为字符串列表

Parameters:
    无
Returns:
    无
Modify:
    2020-04-12
"""


def textParse(bigString):  # 将字符串转换为字符列表
    listOfTokens = re.split(r'\\W', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 除了单个字母，例如大写的I，其它单词变成小写


"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
Modify:
    2020-04-12
"""


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(
            open('email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        # print(wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(
            open('email/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        # print(wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  # 标记非垃圾邮件，1表示垃圾文件
    vocabList = bf.createVocabList(docList)  # 创建词汇表，不重复
    trainingSet = list(range(50))
    # print(vocabList)
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del(trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(
            bf.setOfWords2Vec(
                vocabList,
                docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = of.trainNB0(
        np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = bf.setOfWords2Vec(
            vocabList, docList[docIndex])  # 测试集的词集模型
        if te.classifyNB(np.array(wordVector), p0V, p1V,
                         pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
