# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:39:58 2016

@author: ldy
"""
from skimage import color
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np
from sklearn.decomposition import PCA
import cv2

from detector import *
def train_svm():

    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print np.array(fds).shape,len(labels)


    esitmator = PCA (n_components=300)
    fds = esitmator.fit_transform(fds)

    print np.array(fds).shape
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print "Classifier saved to {}".format(model_path)

    posDir = "./pos"
    negDir = "./neg"
    rootDir = "./neg_person"

    k = 0


    for filename in os.listdir(posDir):

        pathname = os.path.join(posDir, filename)
        print(pathname)

        im = cv2.imread(pathname)
        im = color.rgb2gray(im)
        if des_type == 'HOG':
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                     transform_sqrt=normalize)
        if des_type == 'HOG+PCA':
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                     transform_sqrt=normalize)

        if des_type == 'BOTH+PCA':
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                     transform_sqrt=normalize)
            lbp = local_binary_pattern(im, n_point, radius, 'default')
            lbp = np.reshape(lbp, (2500,))
            fd = np.concatenate((fd, lbp), axis=0)

        print(fd.shape)
        fd = fd.reshape(1, -1)
        fd = esitmator.transform(fd)
        print(fd.shape)
        pred = clf.predict(fd)
        if pred == 1:
            k += 1
        print(pred)



    for filename in os.listdir(negDir):

        pathname = os.path.join(negDir, filename)
        print(pathname)

        im = cv2.imread(pathname)
        im = color.rgb2gray(im)

        if des_type == 'HOG':
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                     transform_sqrt=normalize)
        if des_type == 'HOG+PCA':
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                     transform_sqrt=normalize)
        if des_type == 'BOTH+PCA':
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                     transform_sqrt=normalize)
            lbp = local_binary_pattern(im, n_point, radius, 'default')
            lbp = np.reshape(lbp, (2500,))
            fd = np.concatenate((fd, lbp), axis=0)

        fd = fd.reshape(1, -1)
        fd = esitmator.transform(fd)
        print(fd.shape)
        pred = clf.predict(fd)
        if pred == 0:
            k += 1
        print(pred)

    print(k / 20.0)
    test_folder('test_image', esitmator)

#训练SVM并保存模型
train_svm()
