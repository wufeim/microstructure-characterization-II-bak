#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename : confusion_matrix.py
# @Date : 2020-01-25
# @Author : Wufei Ma

import os
import sys
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Configure matplotlib.
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def plot(confusion_matrix_file, title, clim=None):
    # Load matrix.
    print('Plotting confusion matrix from {:s}...'.format(confusion_matrix_file))
    mat = pd.read_csv(confusion_matrix_file, index_col=0)
    cols = list(mat.columns)

    # Conver to Numpy ndarray.
    mat = mat.to_numpy()

    # Prepare params.
    cmap = plt.get_cmap('Blues')
    thresh = np.nanmax(mat) * 0.6

    # Plot the matrix.
    plt.figure(figsize=(12, 8))
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(clim)

    # Add class names.
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, cols, rotation=45)
    plt.yticks(tick_marks, cols, rotation=45)

    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, '{:.2f}'.format(mat[i, j]),
                 horizontalalignment='center',
                 color='white' if mat[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted')

    output_file = os.path.basename(confusion_matrix_file)[:-4] + '.png'
    output_file = os.path.join('../figures', output_file)
    plt.savefig(output_file, dpi=400)
    print('Plot saved to {:s}.'.format(output_file))


if __name__ == '__main__':

    (filename, title, clim) = ('../results/binary_classification_results_f1.csv', 'Confusion matrix (metric: F1 score)', (0.0, 1.0))
    # (filename, title, clim) = ('../results/binary_classification_results_mcc.csv', 'Confusion matrix (metric: MCC score)', (-1.0, 1.0))

    plot(filename, title, clim)
