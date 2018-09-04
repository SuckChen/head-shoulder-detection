
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np
import cv2

    
def extract_features():
    #des_type = 'LBP'

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        #print im_path
        
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise = visualize, transform_sqrt=normalize)
        if des_type == "HOG+PCA":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise = visualize, transform_sqrt=normalize)

        if des_type == "LBP":
            #fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,transform_sqrt=normalize)
            fd = local_binary_pattern(im, n_point, radius, 'default')
            fd = np.reshape(fd,(2500,))


        if des_type == "BOTH":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,transform_sqrt=normalize)
            lbp = local_binary_pattern(im, n_point, radius, 'default')
            lbp = np.reshape(lbp,(2500,))
            fd = np.concatenate((fd,lbp),axis=0)
        if des_type == "BOTH+PCA":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,transform_sqrt=normalize)
            lbp = local_binary_pattern(im, n_point, radius, 'default')
            lbp = np.reshape(lbp,(2500,))
            fd = np.concatenate((fd,lbp),axis=0)
        if des_type == "HOG+his":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=normalize)
            (b, g, r) = cv2.split(im)
            his0 = cv2.calcHist([b], [0], None, [256], [0.0, 255.0])
            his1 = cv2.calcHist([g], [1], None, [256], [0.0, 255.0])
            his2 = cv2.calcHist([r], [2], None, [256], [0.0, 255.0])
            print(type(his0))
        print(fd.shape)


        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print "Positive features saved in {}".format(pos_feat_ph)

    print "Calculating the descriptors for the negative samples and saving them"
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=normalize)

        if des_type == "HOG+PCA":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise = visualize, transform_sqrt=normalize)

        if des_type == "LBP":
            #fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,transform_sqrt=normalize)
            fd = local_binary_pattern(im, n_point, radius, 'default')
            fd = np.reshape(fd,(2500,))
        if des_type == "BOTH":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,transform_sqrt=normalize)
            lbp = local_binary_pattern(im, n_point, radius, 'default')
            lbp = np.reshape(lbp,(2500,))
            fd = np.concatenate((fd, lbp), axis=0)
        if des_type == "BOTH+PCA":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize,transform_sqrt=normalize)
            lbp = local_binary_pattern(im, n_point, radius, 'default')
            lbp = np.reshape(lbp,(2500,))
            fd = np.concatenate((fd,lbp),axis=0)
        if des_type == "HOG+his":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=normalize)
            b, g, r = cv2.split(im)
            his0 = cv2.calcHist([b], [0], None, [256], [0.0, 255.0])
            his1 = cv2.calcHist([g], [1], None, [256], [0.0, 255.0])
            his2 = cv2.calcHist([r], [2], None, [256], [0.0, 255.0])



        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        #print(type(lbp))
        fd_path = os.path.join(neg_feat_ph, fd_name)
    
        joblib.dump(fd, fd_path)
    print "Negative features saved in {}".format(neg_feat_ph)

    print "Completed calculating features from training images"

if __name__=='__main__':
    extract_features()
