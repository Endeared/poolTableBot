import cv2
import torch
import os
import pandas

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/2-100i-553e-m.pt')
directory = './annotateImgs2'

dir_list = os.listdir(directory)


# for file in dir_list:

#     image = cv2.imread('./annotateImgs2/' + file)
#     results = model(image, size=640)
#     # results.print()
#     # results.save()
#     print(results.pandas().xyxy[0])

image = cv2.imread('./annotateImgs2/' + dir_list[0])
height, width, ch = image.shape
results = model(image, size=640)
# results.print()
# results.save()
cls = results.pandas().xyxy[0]['class']
print(len(cls))
xmin = results.pandas().xyxy[0]['xmin']
ymin = results.pandas().xyxy[0]['ymin']
xmax = results.pandas().xyxy[0]['xmax']
ymax = results.pandas().xyxy[0]['ymax']
conf = results.pandas().xyxy[0]['confidence']

string = ''

for i in range(len(cls)):
    x1 = xmin[i] / width
    x2 = xmin[i] / width
    y1 = ymin[i] / width
    y2 = ymin[i] / width

    valX1 = round(x1, 6)
    valX2 = round(x2 - x1, 6)
    valY1 = round(y1, 6)
    valY2 = round(y2 - y1, 6)

    # print(f'[{cls[i]}] - [{xmin[i]}, {ymin[i]}, {xmax[i]}, {ymax[i]}] - [{conf[i]}]')
    print(f'{cls[i]} {x1} {y1} {x2} {y2} {conf[i]}')
    # string += f'{cls[i]}'