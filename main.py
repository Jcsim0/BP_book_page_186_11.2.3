# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math


# 目标函数
def g(p):
    assert -2 <= p <= 2
    result = 1 + math.sin(math.pi * 0.25 * p)
    # 返回小数点后三位（四舍五入）
    return round(result, 3)


def f1(weight, input_, bias):
    n = np.dot(weight, input_) + bias
    result = 1 / (1 + np.exp(-n))
    return np.round(result, 3)


def f2(weight, input_, bias):
    result = np.dot(weight, input_) + bias
    return np.round(result, 3)


def count_error(target, output):
    return target - output


def forward(error):
    result = -2 * 1 * error
    return np.array([[result]])


def backward(f1_output, w2_T, fowward_):

    # f1_d = (1-(1/(1+np.exp(f1_output)))) * (1/(1+np.exp(f1_output)))
    f1_d = np.array([[ (1-f1_output[0, ])*f1_output[0, ], 0], [0, (1-f1_output[1, ])*f1_output[1, ]]])
    print(f1_d)


if __name__ == '__main__':
    w1 = np.array([[-0.27], [-0.41]])
    a1 = np.array([[1]])
    b1 = np.array([[-0.48], [-0.13]])

    w2 = [[0.09, -0.17]]
    b2 = [[0.48]]

    # print(f2(w2, f1(w1, a1, b1), b2))
    print(f1(w1, a1, b1))
    backward(f1(w1, a1, b1), 0, 0)
