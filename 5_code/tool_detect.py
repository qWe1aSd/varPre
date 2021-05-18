#!/usr/bin/env python
# encoding: utf-8
# @time: 2021/5/16 16:50
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: tool_detect.py
# @software: PyCharm

"""
    six classifiers, svm/rf/knn/mlp/cnn/lstm

"""

import time
import numpy as np
import pandas as pd

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


# get current time
def get_current_time():
    """
    :return: time
    """
    current_time = 'current time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return current_time


# fetch original x and y
def get_x_y(xy):
    """
    from dataframe to ndarray/Series
    :param xy: dataframe
    :return: x ndarray, y Series
    """
    y = xy['appFamily']
    x = xy.copy()
    # delete redundant columns
    del x['appFamily']
    del x['appVariety']
    del x['appName']
    x = x.values  # dataframe to ndarray

    return x, y


# ml-based variant detection
def ml_detection(train_x, test_x, train_y, test_y, classifier, classifier_name, path, logs):
    """
    ml-based variant detection including training, test and saving results
    #
    results:
        logs, training time, test time, confusion matrix and classification report
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param classifier:
    :param classifier_name:
    :param path: output path
    :param logs:
    """
    model_name = classifier_name
    #
    start_time = time.time()
    clf = classifier.fit(train_x, train_y)
    train_time = time.time() - start_time
    #
    start_time = time.time()
    pre_y = clf.predict(test_x)
    test_time = time.time() - start_time

    cm = confusion_matrix(test_y, pre_y, labels=np.unique(test_y))  # (23, 23)
    cr = classification_report(test_y, pre_y, digits=4)

    with open(path, 'a', encoding='utf-8') as f_w:
        f_w.write('\n{}\n'.format(logs))
        f_w.write('\nclassifier: {}\n'.format(model_name))
        f_w.write('\ntrain time (s): {}, test time (s): {}\n'.format(round(train_time, 9), round(test_time, 9)))
        f_w.write('\ntest time (s) for each test app: {}\n'.format(round(test_time / len(test_y), 9)))
        f_w.write('\nconfusion matrix:\n')
        for i in range(cm.shape[0]):
            f_w.write('{}\n'.format(cm.tolist()[i]))
        f_w.write('\nclassification report:\n{}'.format(cr))
        f_w.write('\n---------------------------------------------------------------\n')


# nn-based variant detection
def nn_detection(train_x, test_x, train_y, test_y, classifier, classifier_name, path, logs):
    """
    nn-based variant detection including training, test and saving results
    #
    results:
        logs, training time, test time, confusion matrix and classification report
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param classifier:
    :param classifier_name:
    :param path: output path
    :param logs:
    """
    model_name = classifier_name
    #
    train_y = to_categorical(train_y)  # normal to one hot
    test_y = to_categorical(test_y)  # normal to one hot
    # early stopping
    start_time = time.time()
    classifier.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=150, batch_size=16,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)])
    train_time = time.time() - start_time
    #
    start_time = time.time()
    pre_y = classifier.predict(test_x, batch_size=16, verbose=1)
    test_time = time.time() - start_time

    test_y = np.array([np.argmax(item) for item in test_y])  # one hot to normal
    pre_y = np.array([np.argmax(item) for item in pre_y])  # one hot to normal
    cm = confusion_matrix(test_y, pre_y, labels=np.unique(test_y))  # (23, 23)
    cr = classification_report(test_y, pre_y, digits=4)

    with open(path, 'a', encoding='utf-8') as f_w:
        f_w.write('\n{}\n'.format(logs))
        f_w.write('\nclassifier: {}\n'.format(model_name))
        f_w.write('\ntrain time (s): {}, test time (s): {}\n'.format(round(train_time, 9), round(test_time, 9)))
        f_w.write('\ntest time (s) for each test app: {}\n'.format(round(test_time / len(test_y), 9)))
        f_w.write('\nconfusion matrix:\n')
        for i in range(cm.shape[0]):
            f_w.write('{}\n'.format(cm.tolist()[i]))
        f_w.write('\nclassification report:\n{}'.format(cr))
        f_w.write('\n---------------------------------------------------------------\n')


# variant detection
def var_detection(classifier, classifier_name, path_in, path_out):
    """
    variant detection
    pre-processing dataset, training models and saving results
    :param classifier:
    :param classifier_name:
    :param path_in: input dataset
    :param path_out: output
    """
    # path
    path_train_in = path_in + '/training.csv'
    path_test_in = path_in + '/test.csv'

    # data pre-process
    train_data = pd.read_csv(path_train_in)
    test_data = pd.read_csv(path_test_in)
    train_x, train_y = get_x_y(train_data)
    test_x, test_y = get_x_y(test_data)

    # logs
    logs = 'variant detection | {}'.format(get_current_time())

    # training models and saving results
    if classifier_name in ['svm', 'rf', 'knn']:
        ml_detection(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y,
                     classifier=classifier, classifier_name=classifier_name, path=path_out, logs=logs)
    else:
        nn_detection(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y,
                     classifier=classifier, classifier_name=classifier_name, path=path_out, logs=logs)
