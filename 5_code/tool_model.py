#!/usr/bin/env python
# encoding: utf-8
# @time: 2021/5/16 16:56
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: tool_model.py
# @software: PyCharm

"""
    six classifiers, svm/rf/knn/mlp/cnn/lstm

"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten, LSTM
from keras.optimizers import adam


# svm
def ml_svm():
    return SVC(C=1, decision_function_shape='ovo', gamma=0.001, kernel='linear')


# rf
def ml_rf():
    return RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=20)


# knn
def ml_knn():
    return KNeighborsClassifier(algorithm='brute', n_neighbors=17, weights='distance')


# mlp
def nn_mlp(feature_shape, label_number):
    """
    one layer, MLP
    :param feature_shape:
    :param label_number:
    :return:
    """
    input = Input(shape=(feature_shape,))  # (None, 80)
    x = Dense(units=1024, activation='relu')(input)  # (None, 1024)
    output = Dense(units=label_number, activation='softmax')(x)  # (None, 23)

    mlp = Model(inputs=input, outputs=output)
    mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return mlp


# cnn
def nn_cnn(feature_shape, label_number):
    """
    one layer, CNN
    :param feature_shape:
    :param label_number:
    :return:
    """
    input = Input(shape=(feature_shape,))  # (None, feature_shape)
    x = Embedding(input_dim=feature_shape, output_dim=1024, input_length=feature_shape, trainable=True)(input)
    # (None, feature_shape, 1024)
    x = Conv1D(filters=128, kernel_size=1, activation='relu')(x)  # (None, feature_shape, 128)
    # strides = 1, padding = 'valid', data_format = 'channels_last'
    x = Flatten()(x)  # (None, feature_shape * 128)
    output = Dense(units=label_number, activation='softmax')(x)  # (None, 23)

    cnn = Model(inputs=input, outputs=output)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return cnn


# lstm
def nn_lstm(feature_shape, label_number):
    """
    one layer LSTM
    :param feature_shape:
    :param label_number:
    :return:
    """
    input = Input(shape=(feature_shape,))  # (None, feature_shape)
    x = Embedding(input_dim=feature_shape, output_dim=1024, input_length=feature_shape, trainable=True)(input)
    # (None, feature_shape, 1024)
    x = LSTM(units=128)(x)  # (None, 128)
    output = Dense(units=label_number, activation='softmax')(x)  # (None, 23)

    lstm = Model(inputs=input, outputs=output)
    lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return lstm
