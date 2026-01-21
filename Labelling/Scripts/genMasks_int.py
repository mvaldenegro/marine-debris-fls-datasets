import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

label_intensities = {
                'background'     : 0,
                'bottle'         : 1,
                'can'            : 2,
                'chain'          : 3,
                'drink-carton'   : 4,
                'hook'           : 5,
                'propeller'      : 6,
                'shampoo-bottle' : 7,
                'standing-bottle': 8,
                'tire'           : 9,
                'valve'          : 10,
                'wall'           : 11,
               }

labelPath = '/home/deepak/M__/Code/MarkedData'  
imgPath = '/home/deepak/M__/Data/marine-debris-watertank-release/fls-images'
outPath = '/home/deepak/M__/Code/Masks_int'
outImgPath = '/home/deepak/M__/Code/Imgs'
overlayPath = '/home/deepak/M__/Code/OverLays'
outImgIntPath = '/home/deepak/M__/Code/Masks_Img_Int'

# read the images and generate the integer labels
for mask in os.listdir(labelPath):

    if "json" in mask: # only if mask is a json file
        f = open(os.path.join(labelPath,mask))
        data = json.load(f)
        key = 'imageData'
        del data[key]
#        b = json.dumps(data,indent=4)
        x = data['shapes'] # list of entries

        # read the image
        img_name = data['imagePath']


        img = cv2.imread(os.path.join(imgPath,img_name),0)

        img_new = cv2.resize(img,(320,480),interpolation=cv2.INTER_AREA)

        mask_out = np.zeros((480, 320,3))

        for index in range(len(x)):
            n = x[index]
            label = n['label'] # label
            if label == 'valvev':
                label = 'valve'
            if label == 'bottlw':
                label = 'bottle'
            print(label_intensities[label])
            points = n['points']
            points = np.array(points, np.int32).reshape((-1,1,2))
            cv2.fillPoly(mask_out,[points],label_intensities[label])
        
        cv2.imshow("Mask",mask_out)
        cv2.imshow("img",img_new)
        print(np.unique(mask_out))
        
        np.save(os.path.join(outPath,mask.replace(".json",".npz")),mask_out)
        im_name = mask.replace(".json",".png")
        cv2.imwrite(os.path.join(outImgPath,im_name),img_new)
        cv2.imwrite(os.path.join(outImgIntPath,im_name),mask_out)
        cv2.waitKey(1)