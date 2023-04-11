import mss
import numpy as np
import cv2
import pyautogui
import keyboard
import torch
from functools import partial
from pprint import pprint

scale = 2
height, width = int(720 / scale), int(1280 / scale)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/3-937i-692e-m.pt')

# with mss.mss() as sct:
#     monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

def createTable():

    table = np.zeros((height, width, 3), dtype=np.uint8)
    table[:, :] = (200, 200, 200)
    table = cv2.cvtColor(table, cv2.COLOR_BGR2RGB)

    return table

def showObjects(objects, frame, scale, background=createTable()):
    
    thisFilter = np.ones((3,3), np.uint8)
    diagram = background.copy()
    highestMean = 0
    highestObj = {}
    lowestMean = 1000
    lowestObj = {}

    for object in objects:

        region = 9
        minX = int((object[0] / scale) - region)
        maxX = int((object[0] / scale) + region)
        minY = int((object[1] / scale) - region)
        maxY = int((object[1] / scale) + region)

        frameX1 = int(object[0] - region)
        frameX2 = int(object[0] + region)
        frameY1 = int(object[1] - region)
        frameY2 = int(object[1] + region)

        scaledX = int(object[0] / scale)
        scaledY = int(object[1] / scale)

        crop = frame[frameY1:frameY2, frameX1:frameX2]
        mean = cv2.mean(crop)
        total = 0
        newList = list(mean)
        for value in newList:
            total += value

        if total > highestMean:
            highestMean = total
            highestObj = {
                'object': object,
                'centerX': scaledX,
                'centerY': scaledY
            }
        
        if total < lowestMean:
            lowestMean = total
            lowestObj = {
                'object': object,
                'centerX': scaledX,
                'centerY': scaledY
            }

        diagram = cv2.circle(diagram, (scaledX, scaledY), int(16 / scale), mean, -1)
        diagram = cv2.circle(diagram, (scaledX, scaledY), int(16 / scale), 0, 0)

    diagram = cv2.circle(diagram, (highestObj['centerX'], highestObj['centerY']), int(16 / scale), (255, 255, 255), -1)
    diagram = cv2.circle(diagram, (highestObj['centerX'], highestObj['centerY']), int(16 / scale), 0, 0)

    diagram = cv2.circle(diagram, (lowestObj['centerX'], lowestObj['centerY']), int(16 / scale), 0, -1)
    diagram = cv2.circle(diagram, (lowestObj['centerX'], lowestObj['centerY']), int(16 / scale), 0, 0)

    return diagram

capture = cv2.VideoCapture('./imgs/vid1.mp4')

if (capture.isOpened() == False):
    print('error')

while (capture.isOpened()):
    ret, frame = capture.read()
    keypressed = cv2.waitKey(30)
    if keypressed == ord('q'):
        break

    # img = np.array(sct.grab(monitor))
    detections = model(frame)

    results = detections.pandas().xyxy[0].to_dict()

    coords = []

    for i in range(len(results['xmin'])):
        x1 = int(results['xmin'][i])
        x2 = int(results['xmax'][i])
        y1 = int(results['ymin'][i])
        y2 = int(results['ymax'][i])
        
        halfX = (x2 - x1) / 2
        halfY = (y2 - y1) / 2
        middleX = x1 + halfX
        middleY = y1 + halfY

        center = (int(middleX), int(middleY))

        coords.append([middleX, middleY])


        # cv2.circle(frame, center, 18, (0, 255, 0), -1)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('view', frame)


    diagram = showObjects(coords, frame, scale)
    cv2.imshow('diagram', diagram)
cv2.destroyAllWindows