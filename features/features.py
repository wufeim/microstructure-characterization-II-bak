#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : features.py
# @Date : 2019-09-13
# @Author : Mofii

from __future__ import absolute_import

import os
import sys

import numpy as np
import cv2

from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn import cluster
from scipy.ndimage.filters import median_filter, gaussian_filter

from skimage.feature import local_binary_pattern

import mahotas.features

import utils

rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)
sns.set_style('whitegrid', {'axes.grid': False})


# Kernels for opening and closing.
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel9 = np.ones((9, 9), np.uint8)
kernel11 = np.ones((11, 11), np.uint8)


def haralick_features(names, distance=1):
    f = []
    for i in range(len(names)):
        img = cv2.imread(names[i])
        img = utils.crop_image(img)
        if img is None or img.size == 0 or np.sum(img[:]) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            h = np.zeros((1, 13))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h = mahotas.features.haralick(img, distance=distance, return_mean=True, ignore_zeros=False)
            h = np.expand_dims(h, 0)
        if i == 0:
            f = h
        else:
            f = np.vstack((f, h))
    return f


def lbp_features(names, P=10, R=5):
    f = []
    for i in range(len(names)):
        img = cv2.imread(names[i])
        img = utils.crop_image(img)
        if img is None or img.size == 0 or np.sum(img[:]) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            h = np.zeros((1, 13))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(img, P=P, R=R)
            # h, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
            h, _ = np.histogram(lbp, bins=P+2, range=(0, P+2), density=True)
        if i == 0:
            f = h
        else:
            f = np.vstack((f, h))
    return f


def phase_specific(image_name, d=15, sigma_color=75, sigma_space=75):
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image {:s} cannot be opened."
                                .format(image_name))

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
    p3 = 255 - p3

    connectivity = 8
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(p2, connectivity, cv2.CV_32S)
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
    return f


def phase_specific_feature(image_names, d=15, sigma_color=75, sigma_space=75):
    features = []
    for i in image_names:
        try:
            feature = phase_specific(i, d, sigma_color, sigma_space)
        except FileNotFoundError as e:
            print("File not found ({}): {}".format(e.errno, e.strerror))
            continue
        if len(features) == 0:
            features = feature
        else:
            features = np.vstack((features, feature))

    return features


def segmentation(image_name, d=15, sigma_color=75, sigma_space=75,
                     visualization=True):
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image {:s} cannot be opened."
                                .format(image_name))

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
    p3_feature = np.sum(p3 != 255) / (img.shape[0]*img.shape[1])

    # Apply closing then opening.
    p2 = np.array(res2, copy=True)
    p2[np.where(p2 != min(set(p2.flatten())))] = 255
    p2 = cv2.morphologyEx(p2, cv2.MORPH_OPEN, kernel9)
    p2 = cv2.morphologyEx(p2, cv2.MORPH_CLOSE, kernel3)
    p2[np.where(p3 != 255)] = 255
    p2_feature = np.sum(p2 != 255) / (img.shape[0]*img.shape[1])

    features = np.asarray([1 - p2_feature - p3_feature, p2_feature, p3_feature])

    if visualization:
        colors = [(219 / 255, 94 / 255, 86 / 255),
                  (86 / 255, 219 / 255, 127 / 255),
                  (86 / 255, 111 / 255, 219 / 255)]
        seg_img = np.zeros((img.shape[0], img.shape[1], 3))
        seg_img[(p2 == 255) * (p3 == 255)] = colors[0]
        seg_img[p2 != 255] = colors[1]
        seg_img[p3 != 255] = colors[2]
        return features, seg_img
    else:
        return features


def area_feature(image_names, d=15, sigma_color=75, sigma_space=75):
    features = []
    for i in image_names:
        try:
            feature = segmentation(i, d, sigma_color, sigma_space,
                                   visualization=False)
        except FileNotFoundError as e:
            print("File not found ({}): {}".format(e.errno, e.strerror))
            continue
        if len(features) == 0:
            features = feature
        else:
            features = np.vstack((features, feature))

    return features


def old_segmentation(image_name, median_filter_size=(5, 5, 1), gaussian_sigma=4,
                     visualization=True):
    img = cv2.imread(image_name)
    if img is None:
        raise Exception("Image {:s} cannot be opened.".format(image_name))

    # Pre-process the images.
    img = utils.crop_image(img)
    img = median_filter(img, size=median_filter_size)
    img = gaussian_filter(img, sigma=gaussian_sigma)
    img = img - np.mean(img)
    img = img / np.std(img)

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(img.flatten().reshape(-1, 1))
    l = kmeans.labels_.reshape(img.shape)

    colors = [(219 / 255, 94 / 255, 86 / 255),
              (86 / 255, 219 / 255, 127 / 255),
              (86 / 255, 111 / 255, 219 / 255)]

    seg_img = np.zeros(img.shape)
    f = [(l == 0).sum(), (l == 1).sum(), (l == 2).sum()]
    f = f / sum(f)
    avg = []
    for c in range(3):
        segc = (l == c)
        count = segc.sum()
        avg.append((segc * img[:, :]).sum() / count)
    feature = [x for x, _ in sorted(zip(f, avg), key=lambda pair: pair[1], \
                                    reverse=True)]

    if visualization:
        for i, col in enumerate([x for x, _ in sorted(zip(list(range(3)), \
                                                          avg),
                                                      key=lambda pair: pair[1],
                                                      reverse=True)]):
            segc = (l == col)
            seg_img[:, :] += (segc * (colors[i]))
        return feature, seg_img
    return feature


def old_area_feature(image_names, median_filter_size=(5, 5, 1),
                     gaussian_sigma=4, verbose=True):
    # Contrast limited adaptive histogram equalization.
    # clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    features = []

    # for img_name in image_names:
    for i in tqdm(range(len(image_names))):
        try:
            feature = old_segmentation(image_names[i],
                                       median_filter_size=median_filter_size,
                                       gaussian_sigma=gaussian_sigma,
                                       visualization=False)
        except Exception as e:
            continue
        if len(features) == 0:
            features = feature
        else:
            features = np.vstack((features, feature))

    return features


def phase_count(image_name, phase_id, median_filter_size=(5, 5, 1),
                gaussian_sigma=4, kernel_size=7):
    img = utils.crop_image(cv2.imread(image_name))

    filtered = median_filter(img, size=median_filter_size)
    filtered = gaussian_filter(filtered, sigma=gaussian_sigma)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    normalized = gray - np.mean(gray)
    normalized = normalized / np.std(normalized)

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(normalized.flatten().reshape(-1, 1))
    l = kmeans.labels_.reshape(normalized.shape)
    avg = []
    for i in range(3):
        seg = (l == i)
        count = seg.sum()
        avg.append((seg * gray).sum() / count)
    sorted_l = np.zeros(l.shape)
    sorted_l = sorted_l - 1
    sorted_l[l == np.argmin(avg)] = 2
    sorted_l[l == np.argmax(avg)] = 0
    sorted_l[sorted_l == -1] = 1

    regions = (sorted_l == phase_id).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply opening then closing.
    after_open = cv2.morphologyEx(regions, cv2.MORPH_OPEN, kernel)
    open_first = cv2.morphologyEx(after_open, cv2.MORPH_CLOSE, kernel)

    connectivity = 8
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(open_first, connectivity,
                                         cv2.CV_32S)

    area_mean = np.mean(stats[:, -1])
    area_std = np.std(stats[:, -1])

    # The background is also a label.
    return num_labels - 1, area_mean, area_std


def phase_count_feature(image_names, phase_id):
    features = []

    # for img_name in image_names:
    # for i in tqdm(range(len(image_names))):
    for i in range(len(image_names)):
        feature = phase_count(image_names[i], phase_id=phase_id)
        if len(features) == 0:
            features = feature
        else:
            features = np.vstack((features, feature))

    return np.asarray(features)


if __name__ == '__main__':

    """
    with open('img_names_oct_28.txt') as f:
        raw = f.read()
    all_img_names = raw.strip().split('\n')

    for i in range(20):
        # img_name = '../../data/DUM1154/DUM1154 007 500X 30keV HC14 15mm Left Mid 2 LBE 007.tif'
        img_name = os.path.join('..', np.random.choice(all_img_names))
        img = cv2.imread(img_name)
        f, seg = segmentation(img_name)

        fig = plt.figure(figsize=(6, 3), constrained_layout=True)
        fig.suptitle(os.path.basename(img_name), fontsize=10)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(seg)
        plt.savefig('{}.png'.format(os.path.basename(img_name)[:-4]), dpi=500)
    """

    img = cv2.imread('../../data/DUM1142/DUM1142 015 500X 30keV HC14 Mid Right LBE 015.tif')
    plt.imshow(img)
