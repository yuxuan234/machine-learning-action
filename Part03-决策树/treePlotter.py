#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

"""
函数说明:示例绘制结点测试体验

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无

Modify:
    2020-04-11
"""

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType):
    font = FontProperties(
        fname=r"c:\windows\fonts\simsun.ttc",
        size=14)  # 设置中文字体
    createPlot.ax1.annotate(
        nodeText,
        xy=parentPt,
        xycoords='axes fraction',
        xytext=centerPt,
        textcoords='axes fraction',
        va="center",
        ha="center",
        bbox=nodeType,
        arrowprops=arrow_args,
        FontProperties=font)

"""
函数说明:创建绘制面板

Parameters:
    无
Returns:
    无
Modify:
    2020-04-11
"""
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


if __name__ == '__main__':
    print(createPlot())
