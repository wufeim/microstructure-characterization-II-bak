import numpy as np
import pandas as pd
import cv2

import os
import sys

from matplotlib import pyplot as plt
plt.rcParams['axes.grid'] = False

from scipy.ndimage.filters import median_filter, gaussian_filter

from sklearn import cluster


COLORS = [(219, 94, 86),
          (86, 219, 127),
          (86, 111, 219)]
IMAGE_CSV = 'data-sep-30.csv'


def format_time(time):
    time = int(time)
    if time < 60:
        return '{:d}s'.format(time)
    elif 60 <= time < 3600:
        return '{:d}m {:d}s'.format(time // 60, time % 60)
    elif 3600 <= time:
        return '{:d}h {:d}m {:d}s'.format(time // 3600, (time % 3600) // 60, time % 60)


def crop_image(image):

    if image.shape[0] == 2048 and image.shape[1] == 2560:
        return image[:1920, :]
    elif image.shape[0] == 1428 and image.shape[1] == 2048:
        return image[:1408, :]
    elif image.shape[0] == 1024 and image.shape[1] == 1280:
        return image[:960, :]
    elif image.shape[0] == 1448 and image.shape[1] == 2048:
        return image[:1428, :]
    else:
        raise Exception("Unknown image size: {}".format(image.shape))


def segmentation(image_name, median_filter_size=(5, 5, 1), gaussian_sigma=4):

    img = cv2.imread(image_name)
    if img is None:
        raise Exception("Image {:s} cannot be opened/read.".format(image_name))
    img = crop_image(img)
    img = median_filter(img, size=median_filter_size)
    img = gaussian_filter(img, sigma=gaussian_sigma)
    img = img - np.mean(img)
    img = img / np.std(img)

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(img.flatten().reshape(-1, 1))
    l = kmeans.labels_.reshape(img.shape)

    avg = []
    for c in range(3):
        segc = (l == c)
        count = segc.sum()
        avg.append((segc * img[:, :]).sum() / count)

    seg_img = np.zeros(img.shape, dtype=np.int32)
    for i, col in enumerate([x for x, _ in sorted(zip(list(range(3)),
            avg), key=lambda pair:pair[1], reverse=True)]):
        segc = (l == col)
        seg_img[:, :] += (segc * (COLORS[i]))
    return seg_img


def generate_data(met_id, data_dir, dst):

    rows = pd.read_csv(IMAGE_CSV, index_col=0)
    rows = rows.loc[rows['met_id'] == met_id]
    rows = rows.loc[rows['img_proc'].isin(['LBE', 'LABE'])]
    rows = rows.loc[rows['img_size'].isin(['2048x2560',
                                           '1428x2048',
                                           '1024x1280',
                                           '1448x2048'])]
    img_names = np.asarray(rows['filename'])
    img_names = np.asarray([os.path.join(data_dir, met_id, x)
                            for x in img_names])

    print('Found {:d} images from class {:s}.'.format(len(img_names), met_id))

    assert(os.path.exists(dst))

    for img_name in img_names:
        try:
            cv2.imwrite(os.path.join(dst, os.path.basename(img_name)),
                        segmentation(img_name))
        except:
            raise Exception('Exception raised when applying segmentation to image: {:s}'.format(img_name))


if __name__=='__main__':

    test_img_name = '../data/DUM1142/DUM1142 002 500X 30keV HC14 Center LBE 002.tif'
    test_img = cv2.imread(test_img_name)
    seg_img = segmentation(test_img_name)
    print(np.mean(seg_img))
    print(seg_img[:5, :5, :])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(test_img)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(seg_img)

    plt.show()
