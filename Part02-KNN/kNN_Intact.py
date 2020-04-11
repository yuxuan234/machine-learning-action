import numpy as np
import kNN_Three as kt
import kNN_Two as kw
import kNN_Norm as kn


"""
函数说明:通过输入一个人的三维特征,进行分类输出

Parameters:
    无
Returns:
    无

Modify:
    2020-04-11
"""


def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每周消费的冰激淋公升数："))
    # 打开的文件名
    filename = "datingTestSet2.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = kt.file2matrix(filename)
    # 训练集归一化
    normMat, ranges, minVals = kn.autoNorm(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([precentTats, ffMiles, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classifierResult = kw.classify0(norminArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))


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
    classifyPerson()
