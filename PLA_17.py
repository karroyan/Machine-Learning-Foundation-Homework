# -*- coding:UTF-8 -*-

'''
PLA algorithm for hw1_17 in Machine Learning Foundation
'''

import numpy as np
import re
import argparse
import random

class PLA(object):
    def __init__(self, dim):
        self.__dim = dim


    def data_load(self, filename):
        infile = open(filename, 'r')
        lines = infile.readlines()
        self.__cnt = len(lines)

        x_train = []
        y_train = []
        x = []

        for line in lines:
            x.append(1)
            line_data = re.split('[ \t]', line)
            for data in line_data[:-1]:
                x.append(float(data))
            x_train.append(x)
            y_train.append(int(line_data[-1]))
            x = []

        np_xtrain = np.array(x_train)
        np_ytrain = np.array(y_train)

        infile.close()

        return np_xtrain, np_ytrain


    def perceptron_train(self, filename):
        count = 0
        x_train, y_train = self.data_load(filename)

        for times in range(2000):#repeat 2000 times
            w = np.zeros((self.__dim + 1, 1))#init
            idx_list = [n for n in range(self.__cnt)]
            random.shuffle(idx_list)
            while True:
                flag = 0
                for i in idx_list:#random cycle
                    if np.dot(x_train[i], w) * y_train[i] <= 0:#classify wrong
                        w += 0.5 * y_train[i] * x_train[i].reshape(self.__dim+1, 1)#w = w + 0.5yx
                        count += 1
                        flag = 1
                if flag == 0:
                    break

        return count/2000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "--path", help = "Please input the path of input file.", dest ="path_name")
    args = parser.parse_args()

    perceptron = PLA(4)
    print(perceptron.perceptron_train(args.path_name))
