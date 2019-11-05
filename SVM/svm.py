# 支持向量机 Support Vector Machine
# 凸二次规划问题求解
# @wuhongjun
from numpy import *
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt


# 创建2*num个数据点
def create_data(num):
    # train data
    xn = []
    yn = []
    for i in range(num):
        xn.append([random.random()*40, random.random()*60, 1])
        yn.append(-1.0)
    for i in range(num):
        xn.append([60+random.random()*40, 60+random.random()*40, 1])
        yn.append(1.0)
    print("已随机生成%d个线性可分的数据点" % (num*2))
    return xn, yn


# 绘制数据点
def draw_point(xn):
    # 先解决标题显示不出来的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画点
    x = array(xn)
    middle_index = int(0.5*len(x))
    plt.scatter(x[0:middle_index, 0], x[0:middle_index, 1], color='blue', marker='o', label='+1')
    plt.scatter(x[middle_index:, 0], x[middle_index:, 1], color='red', marker='x', label='-1')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend(loc='upper left')
    plt.title('Data')

# 绘制分类面
def draw_line(wt):
    # 划线
    x1 = 1
    y10 = ((wt[0] * x1 + wt[2]) * (-1) - 1) / wt[1]
    y11 = (wt[0] * x1 + wt[2]) * (-1) / wt[1]
    y12 = ((wt[0] * x1 + wt[2]) * (-1) + 1) / wt[1]
    x2 = 100
    y20 = ((wt[0] * x2 + wt[2]) * (-1) - 1) / wt[1]
    y21 = (wt[0] * x2 + wt[2]) * (-1) / wt[1]
    y22 = ((wt[0] * x2 + wt[2]) * (-1) + 1) / wt[1]
    # print("x1, y1, x2, y2", x1, y1, x2, y2)
    plt.plot([x1, x2], [y10, y20], 'r--')
    plt.plot([x1, x2], [y11, y21], 'black')
    plt.plot([x1, x2], [y12, y22], 'r--')
    # 控制坐标轴范围
    plt.xlim(-10, 100)
    plt.ylim(-10, 100)
    # plt.show()


def draw(xn, wt):
    plt.figure()
    draw_point(xn)
    draw_line(wt)
    plt.show()


# 2*n train data
n = 100
(xn, yn) = create_data(n)

# 求解二次规划 网址https://www.kancloud.cn/wizardforcel/python-quant-uqer/186294
x = matrix(array(xn))
y = matrix(diag(yn))
G = matrix(dot(y, -1.0*x))
P = matrix(diag([1.0] * 3))
q = matrix(zeros((3, 1)))
h = matrix(array([-1.0]*2*n))
solvers.options['show_progress'] = False  # 控制我不感兴趣的输出
sol = solvers.qp(P, q, G, h)
wt = []
for i in sol['x']:
    wt.append(i)
print("wt: ", wt)
draw(xn, wt)











