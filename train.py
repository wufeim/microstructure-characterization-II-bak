#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: train.py
# @Date: 2020-02-16
# @Author: Wufei Ma

import os
import sys

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a classification model.')
    parser.add_argument('model', nargs=1, type=str, metavar='model_type', help='the type of model to be trained (\'binary\' or \'multiclass\')',
                        choices=['binary', 'multiclass'])
    parser.add_argument('data_dir', nargs=1, type=str, metavar='data_dir', help='the directory to the images')
    parser.add_argument('-p', '--plot', action='store_true', help='plot the outputs to figures')
    parser.add_argument('--results_dir', help='specify the results directory', default='results')
    parser.add_argument('--figures_dir', help='specify the figures directory', default='figures')
    args = parser.parse_args()
    print(args)

    print('Starting train.py...')
    # os.environ.update(environ)
