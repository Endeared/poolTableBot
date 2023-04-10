import mss
import numpy as np
import cv2
import pyautogui
import keyboard
import torch
from functools import partial


model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/3-937i-692e-m.pt')

# with mss.mss() as sct:
#     monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

capture = cv2.VideoCapture('./imgs/vid1.mp4')

if (capture.isOpened() == False):
    print('error')



while (capture.isOpened()):
    ret, frame = capture.read()
    cv2.imshow('frame', frame)
    keypressed = cv2.waitKey(30)
    if keypressed == ord('q'):
        break

    # img = np.array(sct.grab(monitor))
    detections = model(frame)

    results = detections.pandas().xyxy[0].to_dict()

    for result in results:
        x1 = int(result['xmin'])
        x2 = int(result['xmax'])
        y1 = int(result['ymin'])
        y2 = int(result['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

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