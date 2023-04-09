# importing opencv, pytorch, os, pandas
import cv2
import torch
import os
import pandas


model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/2-100i-553e-m.pt') # loads yolo model / neural network to identify balls (2-100i-553e-m), 2 = dataset #, 100i = 100 images, 553 = 553 epochs / generations, m = medium model
directory = './annotateImgs2' # directory with fresh 1000 random frames from video

dir_list = os.listdir(directory) # list all png files in directory

for i in range(len(dir_list)): # for item in dir_list
    image = cv2.imread('./annotateImgs2/' + dir_list[i]) # image = .png file
    name = (dir_list[i])[:-4] # name = nameOfFile (no .png at end)
    copy = image.copy() # copy = copy of image
    height, width, ch = copy.shape # pulls height, width from copy file
    results = model(copy, size=640) # applies model to copy file, infers at default size of 640px (higher is better)


    cls = results.pandas().xyxy[0]['class'] # results.pandas().xyxy[0] returns list of details about automatically annotated coordinates, provides 'class' of object, 'xmin, ymin, xmax, ymax' and 'confidence' all in table
    xmin = results.pandas().xyxy[0]['xmin'] # all x / y coordinates are currently in pixel format, not yolo
    ymin = results.pandas().xyxy[0]['ymin']
    xmax = results.pandas().xyxy[0]['xmax']
    ymax = results.pandas().xyxy[0]['ymax']
    conf = results.pandas().xyxy[0]['confidence']

    string = '' # useless

    with open(f'./annotateImgs2/{name}.txt', 'w') as file: # creating a .txt file with the same name stored earlier, has to match .png file name, this is where we will store our yolo coordinates
        for i in range(len(cls)): # for item in each results.pandas table (each object detected)
            x1 = xmin[i] / width # setting converting x and y values to yolo format (proportion between 0 - 1 rather than exact coordinate in pixels)
            x2 = xmax[i] / width
            y1 = ymin[i] / height
            y2 = ymax[i] / height

            valX1 = round(x1, 6) # rounding values
            valX2 = round(x2 - x1, 6)
            valY1 = round(y1, 6)
            valY2 = round(y2 - y1, 6)

            valConf = round(conf[i], 6) # rounding confidence value

            # yolo uses this format for detection: minx, miny, width of annotation, height of annotation
            valHalfXyolo = (x2 - x1)/2 # getting half of width of annotation
            valHalfYyolo = (y2 - y1)/2 # getting half of height of annotation
            valHalfXR = round(valHalfXyolo, 6) # rounding above values
            valHalfYR = round(valHalfYyolo, 6)

            midX = valX1 + (valX2/2) # useless
            midY = valY1 + (valY2/2)

            if valConf > 0.20: # if confidence value > 80%
                file.write(f'15 {valX1 + valHalfXR} {valY1 + valHalfYR} {valHalfXR * 2} {(valHalfYR * 2)}\n') # write file in format: classNumber minx miny width height