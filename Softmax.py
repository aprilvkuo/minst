#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@license: Apache Licence 
@contact: aprilvkuo@gmail.com
@site: 
@software: PyCharm Community Edition
@file: Softmax.py
@time: 2017/11/30 上午12:51
"""
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
import numpy as np
import csv


class Softmax(object):
    """
    传统的softmax
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.sess = tf.InteractiveSession()

    def train(self):
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]), dtype=tf.float32)
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        x_new = tf.layers.batch_normalization(x, training=True)
        y = tf.nn.softmax(tf.matmul(x_new, w) + b)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        # If we don't include the update ops as dependencies on the train step, the
        # tf.layers.batch_normalization layers won't update their population statistics,
        # which will cause the model to fail at inference time
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
        tf.global_variables_initializer().run()

        # next_batch = self.batch()
        next_batch = self.generate_batch()
        for i in range(10000):
            batch_xs, batch_ys = self.sess.run(next_batch)
            # data = next_batch.next()
            # batch_xs, batch_ys = data[0], data[1]
            self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(self.sess.run(accuracy, feed_dict={x: self.features, y_: self.labels}))

    def generate_batch(self):
        """
        可以从numpy文件读取，也可以是一个feedalbe的，用一个session初始化，要用全局，不然sess会被
        当做局部变量，从而使得初始化失败
        :return: 
        """
        features_placeholder = tf.placeholder(self.features.dtype, self.features.shape)
        labels_placeholder = tf.placeholder(self.labels.dtype, self.labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
        dataset = dataset.repeat(100)
        batched_dataset = dataset.batch(100)
        iterator = batched_dataset.make_initializable_iterator()
        self.sess.run(iterator.initializer, feed_dict={features_placeholder: self.features, labels_placeholder: self.labels})
        batch_xs, batch_ys = iterator.get_next()
        return batch_xs, batch_ys

    def generate_batch_2(self,STEPS=10000, BATCH=100):
        """
        通过sklearn接口将训练数据划分得到batch大小的数据，
        然后yield返回。
        :param STEPS: 
        :param BATCH: 
        :return: 
        """
        ss = ShuffleSplit(n_splits=STEPS, train_size=BATCH)
        ss.get_n_splits(self.features, self.labels)
        for step, (idx, _) in enumerate(ss.split(self.features, self.labels), start=1):
            yield self.features[idx], self.labels[idx]



class SoftmaxNew(object):
    """
    用estimator实现
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.features.dtype = 'float64'
        self.labels.dtype = 'float64'

    def model_fn(self, features, labels, mode):
        # 构建一个线性模型并预测值
        W = tf.get_variable("w", [784,10], dtype=tf.float64)
        b = tf.get_variable("b", [10], dtype=tf.float64)
        y = tf.matmul(features['x'], W) + b
        # Loss sub-graph
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
        # Training sub-graph
        global_step = tf.train.get_global_step()


        optimizer = tf.train.GradientDescentOptimizer(0.1)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
        train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1), correct_prediction)

        # EstimatorSpec connects subgraphs we built to the
        # appropriate functionality.
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train)

    def train(self):
        feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": self.features},\
                                                            y=self.labels,num_epochs=1, shuffle=True)

        classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=10,
                                                model_dir="/tmp/iris_model")

        classifier = tf.estimator.Estimator(model_fn=self.model_fn)
        classifier.train(input_fn=train_input_fn, steps=10)
        train_metrics = classifier.evaluate(input_fn=train_input_fn)

        print train_metrics

