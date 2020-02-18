#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename : area_features.py
# @Date : 2020-02-18
# @Author : Wufei Ma

import os
import sys

import argparse

import numpy as np
import pandas as pd

import cv2

import plotly.express as px
import plotly.io as pio

from IPython.display import Image

import utils


def plot_area_features(feature_file, mode, output_filename, scale=4):
    feature_list = ['area_0', 'area_1', 'area_2']
    df = utils.load_features_from_file(feature_file, feature_list)
    if mode == '10_class':
        fig = px.scatter_3d(df, x='area_0', y='area_1', z='area_2',
                            color='met_id')
        fig.update_layout(scene_aspectmode='cube')
        fig.update_layout(title_text='Area Features (10 classes)')
    elif mode == 'binary':
        column1 = ['DUM1178', 'DUM1154', 'DUM1297', 'DUM1144', 'DUM1150', 'DUM1160']
        column2 = ['DUM1180', 'DUM1303', 'DUM1142', 'DUM1148', 'DUM1162']
        labels = df['met_id'].to_numpy().astype(str)
        df['binary_label'] = ['column_1' if x in column1 else 'column_2' for x in labels]
        fig = px.scatter_3d(df, x='area_0', y='area_1', z='area_2',
                            color='binary_label')
        fig.update_layout(scene_aspectmode='cube')
        fig.update_layout(title_text='Area Features (binary)')
    else:
        raise ValueError('Unknown plotting mode. Use \'10_class\' or \'binary\' instead.')
    if output_filename.endswith('.html'):
        pio.write_html(fig, output_filename)
    elif output_filename.endswith('.png'):
        img_str = fig.to_image(format='png', scale=scale)
        arr = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cv2.imwrite(output_filename, img)
    else:
        raise ValueError('Unknown output format given by output_filename: {:s}'.format(output_filename))


if __name__ == '__main__':

    """Example Usage
    feature_file = '../results/all_features.csv'
    plot_area_features(feature_file, '10_class', '../figures/area_features_10_class.png')
    plot_area_features(feature_file, 'binary', '../figures/area_features_binary.png')
    """

    parser = argparse.ArgumentParser(description='Visualization of area features.')
    parser.add_argument(
        'feature_file',
        type=str,
        metavar='feature_file',
        help='the prepared feature file to be plotted'
    )
    parser.add_argument(
        'mode',
        type=str,
        metavar='mode',
        help='the mode to plot the area features, \'10_class\' or \'binary\'',
        choices=['10_class', 'binary']
    )
    parser.add_argument(
        'output_filename',
        type=str,
        metavar='output_filename',
        help='the format and name of the output, \'html\' or \'png\''
    )
    parser.add_argument(
        '--scale',
        type=int,
        metavar='scale',
        help='specify the scale of the output image (ignored when output as HTML)'
    )
    args = parser.parse_args()
    plot_area_features(
        args.feature_file,
        args.mode,
        args.output_filename,
        args.scale
    )
