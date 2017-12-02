#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@license: Apache Licence 
@contact: aprilvkuo@gmail.com
@site: 
@software: PyCharm Community Edition
@file: DataLoader.py
@time: 2017/11/30 上午12:25
"""
import pandas as pd
import numpy as np


class DataLoader():
    def __init__(self, path='./input/train.csv'):
        self.data_path = path
        self.x = None
        self.y = None
        self.new_y = None
        self._load_csv()

    def _load_csv(self):
        data = pd.read_csv(self.data_path)
        self.x = np.asarray(data.iloc[:, 1:])
        self.y = np.asarray(data.iloc[:, 0])
        # one_hot
        new_y = np.zeros([self.y.shape[0], 10])
        new_y[range(len(new_y)), self.y] = 1
        self.y = new_y

    def get_data(self):
        return self.x, self.y