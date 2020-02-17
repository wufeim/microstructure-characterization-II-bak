#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename : binary_classification.py
# @Date : 2019-11-09
# @Author : Wufei Ma

from __future__ import absolute_import

import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import seaborn as sns
from IPython.display import HTML

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


def train_random_forest(X, Y):
    if len(Y.shape) > 1:
        Y = Y.flatten()
    skf = StratifiedKFold(n_splits=5)
    count = 1
    f1_scores = []
    mcc = []
    for train_index, test_index in skf.split(np.zeros((len(Y), 1)), Y):
        count += 1
        train_labels = Y[train_index]
        test_labels = Y[test_index]
        train = X[train_index]
        test = X[test_index]
        model = RandomForestClassifier(n_estimators=100,
                                       max_features='sqrt',
                                       n_jobs=-1, verbose=0)
        model.fit(train, train_labels)
        n_nodes = []
        max_depths = []

        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        train_rf_predictions = model.predict(train)
        train_rf_probs = model.predict_proba(train)[:, 1]
        rf_predictions = model.predict(test)
        rf_probs = model.predict_proba(test)[:, 1]

        if np.sum(test_labels == 0) == 0 or np.sum(test_labels == 1) == 0:
            print('Error: Only one label in test_labels.')
            continue
        if np.sum(rf_predictions == 0) == 0 or np.sum(rf_predictions == 1) == 1:
            print('Error: Only one label in rf_predictions.')
            continue

        # a = sum((rf_predictions == test_labels) / len(test))
        f1_scores.append(f1_score(test_labels, rf_predictions,
                                  average='weighted'))
        mcc.append(matthews_corrcoef(test_labels, rf_predictions))
    return np.mean(f1_scores), np.mean(mcc)


def binary_classification(df, met_ids):
    df = df.loc[df['met_id'].isin(met_ids)]
    f = df[['area_0', 'area_1', 'area_2']].to_numpy()
    l = df['met_id'].to_numpy()
    l = l == met_ids[1]
    if np.sum(l == 0) == 0 or np.sum(l == 1) == 0:
        return np.nan, np.nan
    return train_random_forest(f[:, :2], l)


def train_binary_classification(area_feature_file, output_dir):
    print('Start training a binary classification model...')
    classes = ['DUM1178', 'DUM1154', 'DUM1297', 'DUM1144', 'DUM1150',
               'DUM1160', 'DUM1180', 'DUM1303', 'DUM1142', 'DUM1148']

    print('Loading features from {:s}...'.format(area_feature_file))
    if not os.path.isfile(area_feature_file):
        raise FileNotFoundError('Cannot find area features file {:s}.'.format(area_feature_file))
    df = pd.read_csv(area_feature_file, index_col=0)

    f1_scores = np.zeros((len(classes), len(classes)))
    mcc_scores = np.zeros((len(classes), len(classes)))
    # For each pair of classes
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i > j:
                continue
            elif i == j:
                f1_scores[i, j] = np.nan
                mcc_scores[i, j] = np.nan
                f1_scores[j, i] = np.nan
                mcc_scores[j, i] = np.nan
            else:
                f1_scores[i, j], mcc_scores[i, j] = binary_classification(df, [classes[i], classes[j]])
                f1_scores[j, i], mcc_scores[j, i] = f1_scores[i, j], mcc_scores[i, j]
                print('For class {:s} and {:s}: F1 score: {:.3f}; MCC score: {:.3f}.'
                      .format(classes[i], classes[j], f1_scores[i, j], mcc_scores[i, j]))

    f1_results = pd.DataFrame(f1_scores)
    mcc_results = pd.DataFrame(mcc_scores)
    f1_results.columns = classes
    mcc_results.columns = classes
    f1_results.insert(0, 'class', classes)
    mcc_results.insert(0, 'class', classes)
    f1_results.set_index('class', inplace=True)
    mcc_results.set_index('class', inplace=True)

    f1_results.to_csv(os.path.join(output_dir, 'binary_classification_results_f1.csv'))
    mcc_results.to_csv(os.path.join(output_dir, 'binary_classification_results_mcc.csv'))


if __name__ == '__main__':

    train_binary_classification(
        '../results/all_features.csv',
        '../results'
    )
