# -*- coding:UTF-8 -*-

'''
PLA pocket algorithm for hw1_15 in Machine Learning Foundation
input: file_path
output: times PLA runs to classify right

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

        w = np.zeros((self.__dim + 1, 1))#init
        cur_wrong_count = 0

        for i in range(self.__cnt):
            if np.dot(x_train[i], w) * y_train[i] <= 0:
                cur_wrong_count += 1

        while count < 50:
            i = random.choice(list(range(self.__cnt)))#random cycle
            if np.dot(x_train[i], w) * y_train[i] <= 0:#classify wrong
                w_tmp = w + y_train[i] * x_train[i].reshape(self.__dim+1, 1)
                w_tmp_cnt = 0
                for j in range(self.__cnt):
                    if np.dot(x_train[j], w_tmp) * y_train[j] <= 0:
                        w_tmp_cnt += 1
                if cur_wrong_count > w_tmp_cnt:
                    w = w_tmp
                    cur_wrong_count = w_tmp_cnt
                count += 1

        return w

    def test(self, train_filename, test_filename):
        x_test, y_test = self.data_load(test_filename)
        self.__testcnt = len(y_test)
        count = 0

        for times in range(2000):#repeat 2000 times
            w = self.perceptron_train(train_filename)
            for i in range(self.__testcnt):
                if np.dot(x_test[i], w) * y_test[i] > 0:
                    count += 1
            print(times)

        return count/(2000*self.__testcnt)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "--train_path", help = "Please input the path of train file.", dest ="train_path")
    parser.add_argument("--test", "--test_path", help="Please input the path of test file.", dest="test_path")
    args = parser.parse_args()

    perceptron = PLA(4)
    print(perceptron.test(args.train_path, args.test_path))
