# 支持向量机 Support Vector Machine
# 凸二次规划问题求解
from numpy import *
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt


def draw_point():
    # 先解决标题显示不出来的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画点
    x = array(xn)
    plt.scatter(x[0:200, 0], x[0:200, 1], color='blue', marker='o', label='+1')
    plt.scatter(x[200:, 0], x[200:, 1], color='red', marker='x', label='-1')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend(loc='upper left')
    plt.title('Data')


def draw_line():
    # 划线
    x1 = 1
    y1 = (wt[0]*x1+wt[2])*(-1)/wt[1]
    x2 = 100
    y2 = (wt[0] * x2 + wt[2]) * (-1) / wt[1]
    print("x1, y1, x2, y2",x1, y1, x2, y2)
    plt.plot([x1, x2], [y1, y2], 'black')
    # 控制坐标轴范围
    plt.xlim(-10, 100)
    plt.ylim(-10, 100)
    # plt.show()


# train data

xn = []
yn = []
n = 200
for i in range(n):
    xn.append([random.random()*40, random.random()*40, 1])
    yn.append(-1.0)
for i in range(n):
    xn.append([60+random.random()*40, 60+random.random()*40, 1])
    yn.append(1.0)
print("已随机生成%d个先性可分的数据点，如图point" % (n*2))

# 求解二次规划 网址https://www.kancloud.cn/wizardforcel/python-quant-uqer/186294
# xn = [[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1.5, 1.5, 1],
# [4, 3, 1], [4, 4, 1], [5, 3, 1], [5, 4, 1], [4.5, 3.5, 1]]
x = matrix(array(xn))
#  yn = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
y = matrix(diag(yn))
G = matrix(dot(y, -1.0*x))
# P = matrix(diag([1.0, 1.0, 1.0]))
P = matrix(diag([1.0] * 3))
# q = matrix(array([0.0, 0.0, 0.0]))
q = matrix(zeros((3, 1)))
# h = matrix(array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))
h = matrix(array([-1.0]*2*n))
sol = solvers.qp(P, q, G, h)
wt = []
for i in sol['x']:
    wt.append(i)
print(wt)
draw_point()
draw_line()
plt.show()











