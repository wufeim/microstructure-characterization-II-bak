import numpy as np
import pandas as pd

import cv2

from features.features import *

FEATURES = [
    'phase_count',
    'haralick',
    'lbp',
    'old_area',
    'area',
    'phase_specific'
]


def extract_feature(feature_name, filenames, **kwargs):
    if feature_name not in FEATURES:
        raise Exception('Unknown feature name {:s}.'.format(feature_name))

    if feature_name == 'phase_count':
        try:
            phase_id = kwargs['phase_id']
        except KeyError:
            raise Exception(
                'Trying to extract feature {:s}, but no {:s} specified.'
                .format(feature_name, 'phase_id')
            )
        return phase_count_feature(filenames, phase_id)
    elif feature_name == 'haralick':
        return haralick_features(filenames)
    elif feature_name == 'lbp':
        return lbp_features(filenames)
    elif feature_name == 'old_area':
        return old_area_feature(filenames)
    elif feature_name == 'area':
        return area_feature(filenames)
    elif feature_name == 'phase_specific':
        return phase_specific_feature(filenames)
    """
    elif feature_name == 'all':
        return all_feature(filename)
    """


if __name__ == '__main__':
    pass
