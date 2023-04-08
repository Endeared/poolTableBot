import cv2
import random

vidcap = cv2.VideoCapture("./imgs/vid1.mp4")
# get total number of frames
totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

for i in range(100):
    randomFrameNumber=random.randint(0, totalFrames)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(f'./annotateImgs/randomFrame{i}.png', image)