#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: phase_specific_example.py
# @Date: 2020-02-21
# @Author: Wufei Ma

import os
import sys

import numpy as np
import pandas as pd

import cv2

import utils

# Kernels for opening and closing.
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel9 = np.ones((9, 9), np.uint8)
kernel11 = np.ones((11, 11), np.uint8)


def phase_specific_example(img, basename, d=15, sigma_color=75, sigma_space=75):
    # Pre-process the images.
    img = utils.crop_image(img)
    img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    # Apply KMeans.
    Z = img.astype(np.float32).reshape((-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10,
                                    cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)

    # Apply closing then opening.
    p3 = np.array(res2, copy=True)
    p3[np.where(res2 != min(set(p3.flatten())))] = 255
    p3 = cv2.morphologyEx(p3, cv2.MORPH_CLOSE, kernel9)
    p3 = cv2.morphologyEx(p3, cv2.MORPH_OPEN, kernel9)

    # Apply closing then opening.
    p2 = np.array(res2, copy=True)
    p2[np.where(p2 != min(set(p2.flatten())))] = 255
    p2 = cv2.morphologyEx(p2, cv2.MORPH_OPEN, kernel9)
    p2 = cv2.morphologyEx(p2, cv2.MORPH_CLOSE, kernel3)
    p2[np.where(p3 != 255)] = 255

    p2 = 255 - p2
    cv2.imwrite(basename+'_p2.png', p2)
    p3 = 255 - p3
    cv2.imwrite(basename+'_p3.png', p3)

    connectivity = 8
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(p3, connectivity, cv2.CV_32S)
    """
    f = [
        num_labels - 1,             # number of regions                 0
        np.mean(stats[1:, -1]),     # mean of region areas              1
        np.std(stats[1:, -1]),      # std of region areas               2
        np.std(centroids[1:, 0]),   # std of the x-axis of centroid     3
        np.std(centroids[1:, 1]),   # std of the y-axis of centroids    4
        # mean and std of (area / (width * height))
        np.mean(stats[1:, 4] / (stats[1:, 2] / stats[1:, 3])),        # 5
        np.std(stats[1:, 4] / (stats[1:, 2] / stats[1:, 3]))          # 6
    ]
    """

    """
    connectivity = 8
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(p3, connectivity, cv2.CV_32S)
    f += [
        num_labels - 1,  # number of regions                            7
        np.mean(stats[1:, -1]),  # mean of region areas                 8
        np.std(stats[1:, -1]),  # std of region areas                   9
        np.std(centroids[1:, 0]),  # std of the x-axis of centroid     10
        np.std(centroids[1:, 1]),  # std of the y-axis of centroids    11
        # mean and std of (area / (width * height))
        np.mean(stats[1:, 4] / (stats[1:, 2] * stats[1:, 3])),       # 12
        np.std(stats[1:, 4] / (stats[1:, 2] / stats[1:, 3]))         # 13
    ]
    """
    colors = [(219, 94, 86),
              (86, 219, 127),
              (86, 111, 219)]
    seg_img = np.zeros((img.shape[0], img.shape[1], 3))
    seg_img[(p2 == 0) * (p3 == 0)] = colors[0]
    seg_img[p2 != 0] = colors[1]
    seg_img[p3 != 0] = colors[2]
    cv2.imwrite(os.path.join(basename+'_segmentation.png'), seg_img)
    for i in range(num_labels):
        s = stats[i]
        im = seg_img[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]
        info = 'Region #{} (a/(w*h): {})'.format(i, s[4]/(s[2]*s[3]))
        print(info)
        cv2.imwrite(basename+' '+str(i)+'.png', im)


if __name__ == '__main__':

    img_name = 'DUM1142 015 500X 30keV HC14 Mid Right LBE 015.tif'
    img = cv2.imread(os.path.join('..', 'sample_images', img_name), cv2.IMREAD_GRAYSCALE)
    phase_specific_example(img, os.path.join('..', 'figures', 'phase_specific_examples', img_name[:-4]))
