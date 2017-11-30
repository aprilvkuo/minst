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
from tensorflow.examples.tutorials.mnist import input_data



def generate_batch(features, labels):
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # iterator = dataset.make_initializable_iterator()
    # sess = tf.Session()
    # sess.run(iterator.initializer, feed_dict={features_placeholder: self.features, labels_placeholder: self.labels})
    dataset = dataset.repeat(100)
    batched_dataset = dataset.batch(100)
    iterator = batched_dataset.make_one_shot_iterator()
    batch_xs, batch_ys = iterator.get_next()
    return batch_xs, batch_ys


if __name__ == '__main__':
    dataset = DataLoader.DataLoader()
    features, labels = dataset.get_data()
    #features, _, labels, _ = train_test_split(features, labels, test_size=0.9)
    print features.shape, labels.shape

    # lr = LogisticRegression()
    # print cross_validate(lr,features,labels)

    model = Softmax.Softmax(features, labels)
    model.train()

    # labels = np.arange(0,5000)
    # features = np.zeros([5000,100])
    # features[labels,0] = labels
    # labels = labels.reshape([5000,1])

    # batch = generate_batch(features, labels)
    # sess = tf.Session()
    # for i in range(1000):
    #     batch_xs, batch_ys = sess.run(batch)
    #     print batch_xs, batch_ys
    #     raw_input()

    # sess = tf.Session()
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #
    # for i in range(1000):
    #     # batch_xs, batch_ys = mnist.train.next_batch(100)
    #     batch_xs, batch_ys = mnist.train.next_batch(100)
    #     print batch_xs, batch_ys
    #     raw_input()