#!/usr/bin/env python
# encoding: utf-8
# @time: 2021/5/16 16:27
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: variant_detection.py
# @software: PyCharm


"""
    main file for variant detection

"""

import argparse
import warnings

from tool_model import ml_svm, ml_rf, ml_knn, nn_mlp, nn_cnn, nn_lstm
from tool_detect import var_detection
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# parse arguments
def parse_args():
    """
    parse arguments
    :return:
    """
    try:
        arguments = []
        # parse
        parser = argparse.ArgumentParser()  # 4
        parser.add_argument('-in', nargs=1, type=str, help='str, input path of dataset')
        parser.add_argument('-out', nargs=1, type=str, help='str, output path of result')
        parser.add_argument('-fe', nargs=1, type=int, help='int, number of features')
        parser.add_argument('-fa', nargs=1, type=int, help='int, number of families')
        args = parser.parse_args()
        for key, value in args.__dict__.items():
            arguments.append(value[0])
        return arguments
    except:
        parser = argparse.ArgumentParser()
        parser.print_help()
        raise


# set classifiers
def set_classifier(feature, family):
    """
    set six classifiers
    :param feature: number of feature
    :param family: number of family
    :return:
    """
    svm = ml_svm()
    rf = ml_rf()
    knn = ml_knn()
    mlp = nn_mlp(feature_shape=feature, label_number=family)
    cnn = nn_cnn(feature_shape=feature, label_number=family)
    lstm = nn_lstm(feature_shape=feature, label_number=family)
    #
    cls_ = [svm, rf, knn, mlp, cnn, lstm]
    cls_name_ = ['svm', 'rf', 'knn', 'mlp', 'cnn', 'lstm']
    return [cls_, cls_name_]


# variant detection
def detection(cls, cls_name, path_in, path_out):
    """
    variant detection
    train, predict and save
    :param cls: classifier
    :param cls_name: classifier name
    :param path_in:
    :param path_out:
    :return:
    """
    for _, (cl, cl_name) in enumerate(zip(cls, cls_name)):
        print('\n{}{}{}\n'.format('-' * 20, cl_name, '-' * 20))
        var_detection(classifier=cl, classifier_name=cl_name, path_in=path_in, path_out=path_out)


if __name__ == '__main__':
    # paths, feature, family
    [path_in, path_out, feature_shape, family_number] = parse_args()

    # models
    [classifiers, classifiers_name] = set_classifier(feature=feature_shape, family=family_number)

    # detection
    detection(cls=classifiers, cls_name=classifiers_name, path_in=path_in, path_out=path_out)

    print('.')
