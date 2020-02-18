# Microstructure Characterization II

This is the repository for microstructure characterization research II since May 2019. It is composed of two parts:

- Feature engineering for image classification
- Representation learning with GANs

## Publication

This repo contains code for reproducing key results in [Image driven machine learning based microstructure recognition and quantification on small datasets](#).

Our previous work: [An image-driven machine learning approach to kinetic modeling of a discontinuous precipitation reaction](https://arxiv.org/abs/1906.05496).

## Feature Engineering for Image Classification

### Resources

- ```train.py```
- features
  - ```__init__.py```
  - ```features.py```
- classification
  - ```binary_classification.py```

### Binary Classification

### Visualization

#### Area features

After features are extracted, you can plot the area features by running

```shell script
python plot/area_features.py results/area_featurs.csv binary figures/area_features_binary.png
```

![Area features (10 classes)](figures/area_features_10_class.png)

Run ```python plot/area_features.py -h``` for help. The supported output format are PNG (for static image output) and HTML (for interactive plot).

#### Plot the confusion matrix for binary classification

Before this step, make sure you have trained a binary classification model and have the confusion matrix results ready.

To plot the confusion matrix, simply run

```shell script
python plot/confusion_matrix.py
```

![Confusion matrix](figures/binary_classification_results_f1.png)

The output figure will be saved to the ```./figures``` directory.

## Representation Learning with GANs

### Resources

### System requirements
