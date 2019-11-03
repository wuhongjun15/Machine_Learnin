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
        xn.append([random.random()*60, random.random()*60, 1])
        yn.append(-1.0)
    for i in range(num):
        xn.append([40+random.random()*60, 40+random.random()*60, 1])
        yn.append(1.0)
    print("已随机生成%d个先性可分的数据点" % (num*2))
    return xn, yn


# 求解二次规划最优解，C为松弛变量,返回分类面的法向量wt
# 网址https://www.kancloud.cn/wizardforcel/python-quant-uqer/186294
def svm(x, y, C):
    n = len(x)
    g1 = diag([1.0] * n).tolist()
    g2 = diag([-1.0] * n).tolist()
    G = matrix(r_[g1, g2])  # 矩阵上下拼接,一个控制α上限，一个控制α下限
    # P = matrix(diag([1.0, 1.0, 1.0]))
    P = []
    for i in range(n):
        P.append([])
        for j in range(n):
            P[i].append(y[i] * y[j] * dot(array(x[i]), array(x[j])))
    P = matrix(array(P))
    q = matrix(array([-1.0] * n))
    h1 = array([C] * n).T
    h2 = array([0.0] * n).T
    h = matrix(array(r_[h1, h2]))  # 矩阵上下拼接,一个控制α上限，一个控制α下限
    A = matrix(array(y)).T
    b = matrix(0.0)
    # print("P: ", P, "\n q:", q, "\n G: ", G, "\n h: ", h, "\n A: ", A, "\n b: ", b)
    solvers.options['show_progress'] = False  # 控制我不感兴趣的输出
    sol = solvers.qp(P, q, G, h)

    # 提出α和w
    a = []
    for i in sol['x']:
        if i < 10e-5:  # 减小浮点误差，令10e-5以下的α都为0
            i = 0.0
        a.append(i)
    # print("α：", a,)
    wt = array(zeros((1, 3)))
    for i in range(len(a)):
        wt += float(a[i]) * y[i] * array(x[i])
    wt = wt.tolist()[0]
    # print("wt: ", wt)
    return wt


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


# 数据集性xn，总共分成k份
def test(xn, yn, k, C):
    n = len(xn)
    l = int(n/k)
    count = 0
    for i in range(k):
        if i != k-1:
            test_data_x = xn[i*l:(i+1)*l]
            test_data_y = yn[i*l:(i+1)*l]
        else:
            test_data_x = xn[i*l:]
            test_data_y = yn[i*l:]
        # print("test_data_x", test_data_x)
        # print("test_data_y", test_data_y)
        train_data_x = []+xn[0:i*l]+xn[(i+1)*l:]
        # print("train_data_x: ", train_data_x)
        train_data_y = [] + yn[0:i * l] + yn[(i + 1) * l:]
        # print("train_data_y: ", train_data_y)
        wt = svm(train_data_x, train_data_y, C)
        # draw(xn, wt)
        for i in range(len(test_data_x)):
            if sign(dot(wt, test_data_x[i])) != test_data_y[i]:
                count += 1
    draw(xn, wt)
    print("当前C为：%f, 错误率：%.2f" % (C, count/n))


n = 50
k = 20  # 控制留“一”法的份数
(xn, yn) = create_data(n)
# print("xn:", xn)
# print("yn:", yn)
C_set = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] # 松弛变量 10exp-5 ~ 10exp5
for C in C_set:
    test(xn, yn, k, C)

