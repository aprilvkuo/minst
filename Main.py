#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@license: Apache Licence 
@contact: aprilvkuo@gmail.com
@site: 
@software: PyCharm Community Edition
@file: Main.py
@time: 2017/11/30 上午10:35
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
import DataLoader, Softmax
import numpy as np
import tensorflow as tf






if __name__ == '__main__':
    dataset = DataLoader.DataLoader()
    features, labels = dataset.get_data()
    #features, _, labels, _ = train_test_split(features, labels, test_size=0.9)


    # lr = LogisticRegression()
    # print cross_validate(lr,features,labels)

    model = Softmax.Softmax(features, labels)
    model.train()

