#!/usr/bin/python
# -*- coding: UTF-8 -*-
import feedparser
import numpy as np
import splitdata_five as sd
import bayes_first as bf
import testing_three as tt
import optimize_four as of

"""
函数说明:main函数

Parameters:
    无
Returns:
    无
Modify:
    2020-04-12
"""


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(
        freqDict.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # print(minLen)
    for i in range(minLen):
        wordList = sd.textParse(feed1['entries'][i]['summary'])
        # print(wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = sd.textParse(feed0['entries'][i]['summary'])
        # print(wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bf.createVocabList(docList)  # 创建词汇表
    # print(vocabList)
    top30Words = calcMostFreq(vocabList, fullText)  # 删除前30个单词
    # print(top30Words)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    # print(trainingSet)
    testSet = []  # 创建测试集
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # 训练分类器(get probs) trainNB0
        trainMat.append(bf.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = of.trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # 对剩余项目进行分类
        wordVector = bf.bagOfWords2VecMN(vocabList, docList[docIndex])
        if tt.classifyNB(np.array(wordVector), p0V, p1V,
                         pSpam) != classList[docIndex]:
            errorCount += 1
        print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))
    return vocabList, p0V, p1V


def getTopWords(ny,sf):
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ == '__main__':
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    # print(ny['entries'][0])
    sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
    # print(sf['entries'][0])

    # vocabList, p0V, p1V = localWords(ny, sf)
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # print('vocabList:\n', vocabList)

    getTopWords(ny, sf)


