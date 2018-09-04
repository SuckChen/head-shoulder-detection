import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from config import *
from skimage import color
import matplotlib.pyplot as plt 
import os 
import glob
from skimage.feature import local_binary_pattern
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detector(filename,esitmator):
    im = cv2.imread(filename)
    im = imutils.resize(im, width = min(400, im.shape[1]))
    #min_wdw_sz = (64, 128)
    min_wdw_sz = (50,50)
    step_size = (10, 10)
    downscale = 1.25

    clf = joblib.load(model_path)

    #List to store the detections
    detections = []
    #The current scale of the image 
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale = downscale):
        #The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            if des_type == 'HOG':
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=normalize)
            if des_type == 'HOG+PCA':
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=normalize)
                fd = fd.reshape(1,-1)
                fd = esitmator.transform(fd)

            if des_type == 'LBP':
                fd = local_binary_pattern(im_window, n_point, radius, 'default')
                fd = np.reshape(fd, (2500,))
            if des_type == "BOTH":
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                         transform_sqrt=normalize)
                lbp = local_binary_pattern(im_window, n_point, radius, 'default')
                lbp = np.reshape(lbp, (2500,))
                fd = np.concatenate((fd, lbp), axis=0)
            if des_type == "BOTH+PCA":
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualise=visualize,
                         transform_sqrt=normalize)
                lbp = local_binary_pattern(im_window, n_point, radius, 'default')
                lbp = np.reshape(lbp, (2500,))
                fd = np.concatenate((fd, lbp), axis=0)
                fd = fd.reshape(1, -1)
                fd = esitmator.transform(fd)

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

            if pred == 1:
                
                if clf.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                    int(min_wdw_sz[0] * (downscale**scale)),
                    int(min_wdw_sz[1] * (downscale**scale))))
                 

            
        scale += 1

    clone = im.copy()

    for (x_tl, y_tl, score , w, h) in detections:
        a = float(score)

        print(type(a))
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, score, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print "sc: ", sc
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    #print "shape, ", pick.shape

    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()

def test_folder(foldername,esitmator):

    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        detector(filename,esitmator)

if __name__ == '__main__':
    foldername = 'test_image'
    #test_folder(foldername,esitmator)
