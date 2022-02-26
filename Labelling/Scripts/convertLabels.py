###################################
# convert labels to labelME Format#
###################################

import json
import requests
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os

imgPath = "/home/deepak/MATIAS/Data/marine-debris-watertank-release/fls-images"
outimgPath = '/home/deepak/MATIAS/Code/MarkedData'
labelPath = "/home/deepak/MATIAS/Code/MultiObject_Labels"

orgAnnots = "annotations.json"

annotationFile = os.path.join(imgPath,orgAnnots)
f = open(annotationFile,'r')
data = json.loads(f.read())

#####################
# labelMe parameters#
#####################
version='4.5.4'  # currently installed
flags = {}
shape_type = 'polygon'
group_id = None
imagePath = ""
shape = []

#####################################
# create the output json file
#{'version':"4.5.4",'flags':{},
# 'shapes':[{'label':'Can',
# 'Points':[[x1,y1],[x2,y2]], 
# 'groups':null,
# "shape_type":"polygon"}], 
# 'imagePath':}
######################################

font = cv2.FONT_HERSHEY_SIMPLEX

for im in os.listdir(imgPath):
    if ".png" in im:
        rect = []
        img = cv2.imread(os.path.join(imgPath,im),0)
        shapes = []
        #####################
        # Converting formats#
        #####################

        labelData = data[im]               # this stores the label corresponding to the image
        boundingBox = labelData['bounding-boxes']

        ####################################    
        # check the no. of labels
        # then assign a label to each class
        ####################################

        for i in range(len(boundingBox)):
            label = boundingBox[i]
            x,y,w,h = label['top-left-x'],label['top-left-y'],label['width'], label['height']        
            rect = [[x,y],[x+w,y+h]]
            class_name = label['class']
            label_ = {'label':class_name, 'points':rect, 'group_id':group_id, 'shape_type':shape_type}
            shapes.append(label_)
            img = cv2.rectangle(img, (rect[0][0],rect[0][1]), (rect[1][0], rect[1][1]), 255, 2)
            cv2.putText(img,class_name,(rect[0][0],rect[0][1]), font, 1,255,1)

            ## Put label on the image


        #########################################################################################
        #label = boundingBox[0]
        #x,y,w,h = label['top-left-x'],label['top-left-y'],label['width'], label['height']
        #rect = [[x,y],[x+w,y+h]]           # top left and bottom right coordinates
        #class_name = label['class']

        #label = {'label':class_name, 'points':rect, 'group_id':group_id, 'shape_type':shape_type}
        #shapes = [label]
        ##########################################################################################

        a = {'version':version, 'flags':{}, 'shapes':shapes, 'imagePath':im}

        b = json.dumps(a, indent=4)     
        print(b)        
        print("\n")
        
        outFile = im.replace(".png",".json")
        print(outFile)

        ####################################################################################
        #with open(outFile, "w") as outfile: 
        #    outfile.write(b) 

        #img = cv2.rectangle(img, (rect[0][0],rect[0][1]), (rect[1][0], rect[1][1]), 255, 3)
        ####################################################################################
        with open(outFile, "w") as outfile: 
            outfile.write(b) 
        cv2.imshow("Image",img)
        cv2.imwrite(os.path.join(outimgPath,im),img)
        cv2.waitKey(1)