# -*- coding: UTF-8 -*-

from numpy import *
import matplotlib.pyplot as plt
import time


# 统计w的错误点，返回为在x里的下标集合
def count_wrong_point(x, y, w):
    wrong_point = []
    for i in range(len(x)):
        if sign(dot(array(x[i]), array(w))) != y[i]:
            wrong_point.append(i)
    return wrong_point


def draw_point():
    # 先解决标题显示不出来的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画点
    x = array(xn)
    plt.scatter(x[1:100, 0], x[1:100, 1], color='blue', marker='o', label='+1')
    plt.scatter(x[100:, 0], x[100:, 1], color='red', marker='x', label='-1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
    plt.title('Data')


def draw_line():
    # 划线
    x1 = 0
    y1 = (wt[0]*x1+wt[2]*(-1))*(-1)/wt[1]
    x2 = 100
    y2 = (wt[0] * x2 + wt[2] * (-1)) * (-1) / wt[1]
    plt.plot([x1, x2], [y1, y2], 'black')
    # 控制坐标轴范围
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    # plt.show()


def draw():
    draw_point()
    draw_line()


# training data
xn = []
yn = []
n = 100
for i in range(n):
    xn.append([random.random()*40, random.random()*60, -1])
    yn.append(-1)
for i in range(n):
    xn.append([60+random.random()*40, 40+random.random()*60, -1])
    yn.append(1)
print("已随机生成%d个先性可分的数据点，如图point" % (n*2))
plt.figure("Point")
draw_point()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~pla~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# wt[0]代表xn[0]的比重，wt[1]代表xn[1]的比重，wt[2]代表threshold
# 初始化wt为0向量
wt = [1, 1, 1]


count = 0
time_start = time.time()
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
time_end = time.time()
total_time = time_end - time_start
# 第二次绘制 wt已经迭代
print("PLA算法经过 %d 次迭代，用时 %f 秒，最后w为: %s,分割面如图PLA所示" % (count, total_time, wt))
plt.figure("PLA")
draw()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~pocket~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
wt = [1, 1, 1]
# w0用来存储在迭代过程中最优的w，初始为wt
w0 = [1, 1, 1]
wt_wrong_point = count_wrong_point(xn, yn, wt)
w0_wrong_point = count_wrong_point(xn, yn, w0)
# 用count控制迭代次数，当选到某一w时，再迭代count次时还未找到更优的w就结束迭代，或者将点全部分开的时候结束迭代
count = 0
t = 0
time_start = time.time()
# 既满足线性可分集又满足非线性可分集，但注意对于线性可分集要让count足够大才好
while count < 8888 and len(wt_wrong_point) != 0:
    t += 1
    # 随机选取其中某一错误点，进行优化
    i_w = random.choice(wt_wrong_point)
    # 比较优化后的wn和wt的优劣
    wn = wt + dot(array(xn[i_w]), array(yn[i_w]))
    wn_wrong_point = count_wrong_point(xn, yn, wn)
    # print("w0: ", w0, "wt: ", wt)
    # print("len(w0_wrong_point): ", len(w0_wrong_point), "len(wt_wrong_point): ", len(wt_wrong_point))
    if len(wn_wrong_point) < len(w0_wrong_point):
        w0 = []
        for i in wn:
            w0.append(i)
        w0_wrong_point = count_wrong_point(xn, yn, w0)
        count = 0
    else:
        count += 1
    wt = wn
    wt_wrong_point = count_wrong_point(xn, yn, wt)
time_end = time.time()
total_time = time_end - time_start
wt = w0
print("Pocket算法经过 %d 次迭代，用时 %f 秒，最后w为: %s，分割面如图Pocket所示。" % (t, total_time, wt))
plt.figure("Pocket")
draw()
plt.show()

