# -*- coding: UTF-8 -*-

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


xn = []
yn = []
for i in range(50):
    xn.append([random.random()*40, random.random()*40, -1])
    yn.append(-1)
for i in range(50):
    xn.append([50+random.random()*40, 50+random.random()*40, -1])
    yn.append(1)
# wt[0]代表xn[0]的比重，wt[1]代表xn[1]的比重，wt[2]代表threshold
# 初始化wt为0向量
wt = [1, 1, 1]
# 第一次绘制 wt还未迭代
# draw()

count = 0
while True:
    count += 1
    complete = True
    for i in range(len(xn)):
        if sign(dot(array(xn[i]), array(wt))) != yn[i]:
            complete = False
            wt = wt + dot(array(xn[i]), array(yn[i]))
            break
    if complete:
        break

# 第二次绘制 wt已经迭代
print(count)
draw()
print(wt)
print("haha")
