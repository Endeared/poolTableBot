import mss
import numpy as np
import cv2
import pyautogui
import keyboard
import torch
from functools import partial


model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/2-100i-553e-m.pt')

with mss.mss() as sct:
    monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

while True:
    img = np.array(sct.grab(monitor))
    results = model(img)

    rl = results.xyxy[0].tolist()

    if len(rl) > 0:
        if rl[0][4] > .05:
            print('found')
            if rl[0][5] == 0:
                x = int(rl[0][2])
                y = int(rl[0][3])
                width = int(rl[0][2] - rl[0][0])
                height = int(rl[0][3] - rl[0][1])

                xpos = int(.10 * ((x - (width/2)) - pyautogui.position()[0]))
                ypos = int(.10 * ((y - (height/2)) - pyautogui.position()[1]))
                extra = [xpos, ypos]

    cv2.imshow('s', np.squeeze(results.render()))
    cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        break
cv2.destroyAllWindows