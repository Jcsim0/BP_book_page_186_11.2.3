# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created with PyCharm
@Auth Jcsim
@Date 2020-12-9 20:10
@File main.py
@Description
"""
import random
import matplotlib.pyplot as plt
import numpy as np
import math

# 保留小数点后几位
accuracy = 3
# 学习率
lr = 0.1


# 目标函数
def g(p):
    assert -2 <= p <= 2
    result = 1 + math.sin(math.pi * 0.25 * p)
    # 返回小数点后三位（四舍五入）
    return round(result, accuracy)


# 得出输出a1
def f1(weight, input_, bias):
    n = np.dot(weight, input_) + bias
    result = 1 / (1 + np.exp(-n))
    return np.round(result, accuracy)


# 得出输出a2
def f2(weight, input_, bias):
    result = np.dot(weight, input_) + bias
    return np.round(result, accuracy)


# 计算误差
def count_error(p, output):
    return g(p) - output


def forward(error_):
    # 计算第二层敏感度
    result = -2 * 1 * error_
    return np.array([[result]])


def backward(f1_output, w2_T, fowward_):
    # 计算f1 导数
    f1_d = np.round(np.subtract(1, f1_output)*f1_output, accuracy)
    # 格式化 f1导数
    f1_d_d = np.array([[f1_d[0][0], 0],
                       [0, f1_d[1][0]]])
    # print(f1_d_d)
    # 第一层敏感度
    result = np.dot(np.dot(f1_d_d, w2_T), fowward_)
    # print(f1_d_d)
    # print(w2_T)
    # print(fowward_)
    return np.round(result, 4)


def data():
    train_data = []
    for i in range(0, 401):
        train_data.append(round(-2.0 + i * 0.01, 2))
    # print(train_data)
    #  随机数据
    random.shuffle(train_data)

    return train_data


if __name__ == '__main__':
    # 初始的权重和偏置
    w1 = np.array([[-0.27], [-0.41]])
    a1 = np.array([[1]])
    b1 = np.array([[-0.48], [-0.13]])
    w2 = np.array([[0.09, -0.17]])
    b2 = np.array([[0.48]])
    train_data = data()
    for epoch in range(3):
        for i in range(0, 401):
            x = train_data[i]
            # 将输入格式化
            input_ = np.array([[x]])
            # 得出a1
            f1_a = f1(w1, input_, b1)
            # 得出a2
            f2_a = f2(w2, f1_a, b2)
            # 计算误差
            error = round(count_error(x, f2_a)[0][0], accuracy)
            # 前向传播，计算s2
            s2 = forward(error)
            # 利用s2 反向传播计算s1
            s1 = backward(f1_a, w2.T, s2)
            # 更新权重和偏置
            w2 = np.subtract(w2, lr * np.dot(s2, f1_a.T))
            b2 = np.subtract(b2, lr * s2)
            w1 = np.subtract(w1, lr * np.dot(s1, input_.T))
            b1 = np.subtract(b1, lr * s1)
            print("Epoch:", epoch, "-", i, ":", error)

    # 在[-2, 2]范围内计算，返回100个均匀间隔的样本。 list类型
    xx = np.linspace(-2, 2, 100)
    # y轴数据
    y_list = []
    for x in xx:
        input_ = np.array([[x]])
        f1_a = f1(w1, input_, b1)
        f2_a = f2(w2, f1_a, b2)
        y_list.append(f2_a[0][0])
    # 生成训练出来的曲线
    plt.plot(xx, y_list, 'r')

    x_data = []
    y_data = []
    for i in range(0, 401):
        x_data.append(-2.0 + i * 0.01)
        y_data.append(g(-2.0 + i * 0.01))
    # 生成目标曲线
    plt.plot(x_data, y_data)
    plt.show()

    # input_ = np.array([[1]])
    # f1_a = f1(w1, input_, b1)
    # f2_a = f2(w2, f1_a, b2)
    # error = count_error(1, f2_a)
    # s2 = forward(error)
    # s1 = backward(f1_a, w2.T, s2)
    # print(s1)
    # print((g(1)-f2_a)[0][0])
    # print(f2(w2, f1(w1, a1, b1), b2))
    # print(forward(1.261))
    #
    # result = f1(w1, a1, b1)
    # print(backward(f1(w1, a1, b1), w2.T, forward(1.261)))
