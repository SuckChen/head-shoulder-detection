
import  os
import cv2

import numpy as np
rootDir = "./clear"
i=1237
for filename in os.listdir(rootDir):

    pathname = os.path.join(rootDir, filename)
    print(pathname)

    im = cv2.imread(pathname)
    cv2.imshow("s",im)
    #cv2.waitKey(0)
    for j in range(20):
        r = cv2.selectROI(im)

        # Crop image
        im1 = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        im1 = cv2.resize(im1, (50, 50),
                         interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./data/images/NEG2/neg-"+str(i)+".jpg",im1)
        i+=1

'''
rootDir = "./pos_person/"
i=1
im  = cv2.imread("./pos_person/new/83.jpg")
a=im.shape
p1=cv2.resize(im,(100,100),

               interpolation=cv2.INTER_CUBIC)
cv2.imshow("s",p1)
cv2.waitKey(0)

for filename in os.listdir(rootDir):

    pathname = os.path.join(rootDir, filename)
    print(pathname)

    im = cv2.imread(pathname)
    cv2.imshow("s",im)
    #cv2.waitKey(0)
    r = cv2.selectROI(im)

    # Crop image
    im1 = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    cv2.imwrite("./pos_person/new/"+str(i)+".jpg",im1)
    i+=1
'''