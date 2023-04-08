import cv2
import numpy as np

image = cv2.imread('./imgs/image1.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
kernel = np.ones((7,7), np.uint8)

lower_bound = np.array([100, 230, 230])
upper_bound = np.array([110, 255, 255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)

cv2.imshow("Image", mask)

cv2.waitKey(0)
cv2.destroyAllWindows