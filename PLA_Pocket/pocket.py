# -*- coding: UTF-8 -*-
# @wuhongjun
from numpy import *
import matplotlib.pyplot as plt


def draw():
    # 先解决标题显示不出来的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画点
    x = array(xn)
    plt.scatter(x[1:50, 0], x[1:50, 1], color='blue', marker='o', label='+1')
    plt.scatter(x[50:, 0], x[50:, 1], color='red', marker='x', label='-1')
    plt.xlabel('收入')
    plt.ylabel('消费')
    plt.legend(loc='upper left')
    plt.title('训练数据')
    # 划线
    x1 = 0
    y1 = (wt[0]*x1+wt[2]*(-1))*(-1)/wt[1]
    x2 = 100
    y2 = (wt[0] * x2 + wt[2] * (-1)) * (-1) / wt[1]
    plt.plot([x1, x2], [y1, y2], 'r')
    plt.show()


# 统计w的错误点，返回为在x里的下标集合
def count_wrong_point(x, y, w):
    wrong_point = []
    for i in range(len(x)):
        if sign(dot(array(x[i]), array(w))) != y[i]:
            wrong_point.append(i)
    return wrong_point


xn = []
yn = []
for i in range(50):
    xn.append([random.random()*60, random.random()*60, -1])
    yn.append(-1)
for i in range(50):
    xn.append([40+random.random()*60, 40+random.random()*60, -1])
    yn.append(1)
# wt[0]代表xn[0]的比重，wt[1]代表xn[1]的比重，wt[2]代表threshold
# 初始化wt为0向量
wt = [1, 1, 1]
# w0用来存储在迭代过程中最优的w，初始为wt
w0 = [1, 1, 1]
wt_wrong_point = count_wrong_point(xn, yn, wt)
w0_wrong_point = count_wrong_point(xn, yn, w0)
# 用count控制迭代次数，当迭代次数达到count时还未找到更优的w就结束迭代，或者将点全部分开的时候结束迭代
count = 0
while count < 888 and len(wt_wrong_point) != 0:
    # 随机选取其中某一错误点，进行优化
    i_w = random.choice(wt_wrong_point)
    # 比较优化后的wn和wt的优劣
    wn = wt + dot(array(xn[i_w]), array(yn[i_w]))
    wn_wrong_point = count_wrong_point(xn, yn, wn)
    # print("w0: ", w0, "wt: ", wt)
    # print("len(w0_wrong_point): ", len(w0_wrong_point), "len(wt_wrong_point): ", len(wt_wrong_point))
    if len(wn_wrong_point) <= len(w0_wrong_point):
        w0 = []
        for i in wn:
            w0.append(i)
        w0_wrong_point = count_wrong_point(xn, yn, w0)
        count = 0
    else:
        count += 1
    wt = wn
    wt_wrong_point = count_wrong_point(xn, yn, wt)

wt = w0
# 第二次绘制 wt已经迭代
draw()
print(wt)

