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

for i in range(len(dir_list)):
    image = cv2.imread('./annotateImgs2/' + dir_list[i])
    name = (dir_list[i])[:-4]
    copy = image.copy()
    print(name)
    height, width, ch = copy.shape
    results = model(copy, size=640)
    # results.print()
    # results.save()
    cls = results.pandas().xyxy[0]['class']
    xmin = results.pandas().xyxy[0]['xmin']
    ymin = results.pandas().xyxy[0]['ymin']
    xmax = results.pandas().xyxy[0]['xmax']
    ymax = results.pandas().xyxy[0]['ymax']
    conf = results.pandas().xyxy[0]['confidence']

    string = ''

    with open(f'./annotateImgs2/{name}.txt', 'w') as file:
        for i in range(len(cls)):
            x1 = xmin[i] / width
            x2 = xmax[i] / width
            y1 = ymin[i] / height
            y2 = ymax[i] / height

            valX1 = round(x1, 6)
            valX2 = round(x2 - x1, 6)
            valY1 = round(y1, 6)
            valY2 = round(y2 - y1, 6)

            valConf = round(conf[i], 6)

            valHalfX = xmax[i] - xmin[i]
            valMidX = xmin[i] + valHalfX/2
            valHalfY = ymax[i] - ymin[i]
            valMidY = ymin[i] + valHalfY/2

            valHalfXyolo = (x2 - x1)/2
            valHalfYyolo = (y2 - y1)/2
            valHalfXR = round(valHalfXyolo, 6)
            valHalfYR = round(valHalfYyolo, 6)

            print(f'{cls[i]} {valX1} {valY1} {valX2} {valY2} {valConf}')
            midX = valX1 + (valX2/2)
            midY = valY1 + (valY2/2)
            if valConf > 0.80:
                file.write(f'{cls[i]} {valX1 + valHalfXR} {valY1 + valHalfYR} {valHalfXR * 2} {(valHalfYR * 2)}\n')
                # cv2.circle(copy, (int(valMidX), int(valMidY)), 8, (0, 0, 255), -1)
                # cv2.imshow('copy', copy)
                # cv2.waitKey(0)