import mss
import numpy as np
import cv2
import pyautogui
import keyboard
import torch
from functools import partial
from pprint import pprint

height, width = 500, 1000
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/3-937i-692e-m.pt')

# with mss.mss() as sct:
#     monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

def createTable():

    table = np.zeros((height, width, 3), dtype=np.uint8)
    table[:, :] = (200, 200, 200)
    table = cv2.cvtColor(table, cv2.COLOR_BGR2RGB)

    return table

def showObjects(objects, background=createTable())
    
    

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

        cv2.circle(frame, center, 12, (0, 255, 0), -1)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('diagram', frame)

    # if len(rl) > 0:
    #     if rl[0][4] > .05:
    #         print('found')
    #         if rl[0][5] == 0:
    #             x = int(rl[0][2])
    #             y = int(rl[0][3])
    #             width = int(rl[0][2] - rl[0][0])
    #             height = int(rl[0][3] - rl[0][1])

    #             xpos = int(.10 * ((x - (width/2)) - pyautogui.position()[0]))
    #             ypos = int(.10 * ((y - (height/2)) - pyautogui.position()[1]))
    #             extra = [xpos, ypos]

    # cv2.imshow('s', np.squeeze(results.render()))
    # cv2.waitKey(1)
    # if keyboard.is_pressed('q'):
    #     break
cv2.destroyAllWindows