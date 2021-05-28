# visualize semantic labels
# overlap the labels on the grayscale images with transparency

import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

# set the paths to the images and labels

pathImgs = input("Enter path for images --> ")
pathLabels = input("Enter path for labels --> ")


# labels and corresponding intensity values
label_intensities = {
                0: 'background'     ,    
                1: 'bottle'         ,
                2: 'can'            ,
                3: 'chain'          ,
                4: 'drink-carton'   ,
                5: 'hook'           ,
                6: 'propeller'      ,
                7: 'shampoo-bottle' ,
                8: 'standing-bottle',
                9: 'tire'           ,
                10: 'valve'         ,
                11: 'wall'           
               }



for im in os.listdir(pathImgs):
    img = cv2.imread(os.path.join(pathImgs, im)) # read images as RGB so as to show colored labels
    label = cv2.imread(os.path.join(pathLabels,im),0) # read as grayscale image

    # put color on the labelled part
    for row in range(label.shape[0]):
        for col in range(label.shape[1]):
            val = label[row, col]
            clrs = img[row, col]
            if val != 0:
                img[row, col] = np.array([clrs[0], clrs[1], val])
    
    # naming the segmented part
    labelCpy = label.copy()
    cnts = cv2.findContours(labelCpy, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ## calculate centroid of each segmented object and name accordingly
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        labelInt = label[cY, cX]

        # draw the contour and center of the shape on the image
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.putText(img, label_intensities[labelInt], (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    
    cv2.imshow("image with label", img)
    cv2.waitKey(0)
    cv2.imwrite("img.png", img)