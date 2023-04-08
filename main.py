import cv2
import numpy as np
import matplotlib.pyplot as plot

def draw_rectangles(ctrs, img):
    
    output = img.copy()
    
    for i in range(len(ctrs)):
    
        M = cv2.moments(ctrs[i]) # moments
        rot_rect = cv2.minAreaRect(ctrs[i])
        w = rot_rect[1][0] # width
        h = rot_rect[1][1] # height
        
        box = np.int64(cv2.boxPoints(rot_rect))
        cv2.drawContours(output,[box],0,(255,100,0),2) # draws box
        
    return output

def filter_contours(ctrs, min_s = 90, max_s = 358, alpha = 3.445):  
    
    filtered_ctrs = [] # list for filtered contours
    
    for x in range(len(ctrs)): # for all contours
        
        rot_rect = cv2.minAreaRect(ctrs[x]) # area of rectangle around contour
        w = rot_rect[1][0] # width of rectangle
        h = rot_rect[1][1] # height
        area = cv2.contourArea(ctrs[x]) # contour area 

        
        if (h*alpha<w) or (w*alpha<h): # if the contour isnt the size of a snooker ball
            continue # do nothing
            
        if (area < min_s) or (area > max_s): # if the contour area is too big/small
            continue # do nothing 

        # if it failed previous statements then it is most likely a ball
        filtered_ctrs.append(ctrs[x]) # add contour to filtered cntrs list

        
    return filtered_ctrs # returns filtere contours

def find_ctrs_color(ctrs, input_img):

    K = np.ones((3,3),np.uint8) # filter
    output = input_img.copy() #np.zeros(input_img.shape,np.uint8) # empty img
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) # gray version
    mask = np.zeros(gray.shape,np.uint8) # empty mask

    for i in range(len(ctrs)): # for all contours
        
        # find center of contour
        M = cv2.moments(ctrs[i])
        cX = int(M['m10']/M['m00']) # X pos of contour center
        cY = int(M['m01']/M['m00']) # Y pos
    
        mask[...]=0 # reset the mask for every ball 
    
        cv2.drawContours(mask,ctrs,i,255,-1) # draws the mask of current contour (every ball is getting masked each iteration)

        mask =  cv2.erode(mask,K,iterations=3) # erode mask to filter green color around the balls contours
        
        output = cv2.circle(output, # img to draw on
                         (cX,cY), # position on img
                         20, # radius of circle - size of drawn snooker ball
                         cv2.mean(input_img,mask), # color mean of each contour-color of each ball (src_img=transformed img)
                         -1) # -1 to fill ball with color
    return output

image = cv2.imread('./imgs/image1.jpg')

height, width = 560, 280

coordsImage = {
    'tl': (169, 110),
    'tla': [169, 110],
    'bl': (168, 335),
    'bla': (168, 335),
    'tr': (633, 108),
    'tra': (633, 108),
    'br': (635, 335),
    'bra': (635, 335)
}

coordsDiagram = {
    'tl': (0, height),
    'tla': [0, height],
    'bl': (0, 0),
    'bla': [0, 0],
    'tr': (width, height),
    'tra': [width, height],
    'br': (width, 0),
    'bra': [width, 0]
}

table = image.copy()
cv2.circle(table, coordsImage['tl'], 8, (0, 0, 255), -1) # top left pocket
cv2.circle(table, coordsImage['bl'], 8, (0, 0, 255), -1) # bot left pocket
cv2.circle(table, coordsImage['tr'], 8, (0, 0, 255), -1) # top right pocket
cv2.circle(table, coordsImage['br'], 8, (0, 0, 255), -1) # bot right pocket

newImage = np.zeros((height, width, 3), dtype=np.uint8)
finalImage = newImage.copy()
cv2.circle(finalImage, coordsDiagram['tl'], 8, (0, 0, 255), -1) # top left pocket
cv2.circle(finalImage, coordsDiagram['bl'], 8, (0, 0, 255), -1) # bot left pocket
cv2.circle(finalImage, coordsDiagram['tr'], 8, (0, 0, 255), -1) # top right pocket
cv2.circle(finalImage, coordsDiagram['br'], 8, (0, 0, 255), -1) # bot right pocket

firstPoints = np.float32([coordsImage['tla'], coordsImage['bla'], coordsImage['tra'], coordsImage['bra']])
secondPoints = np.float32([coordsDiagram['tla'], coordsDiagram['bla'], coordsDiagram['tra'], coordsDiagram['bra']])
matrix = cv2.getPerspectiveTransform(firstPoints, secondPoints)
warped = cv2.warpPerspective(image, matrix, (width, height))


hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
kernel = np.ones((7,7), np.uint8)

lower_bound = np.array([100, 230, 230])
upper_bound = np.array([110, 255, 255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)
mask_others = cv2.bitwise_not(mask)
mask = cv2.bitwise_and(warped, warped, mask=mask_others)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
detected_objects = draw_rectangles(contours, warped)
contours_final = filter_contours(contours)
detected_objects_final = draw_rectangles(contours_final, warped)

contours_color = find_contours_color(contours_final, warped)
contours_color = cv2.addWeighted(contours_color, 0.5, warped, 0.5, 0)


cv2.imshow("Image", table)
cv2.imshow('Image2', warped)
cv2.imshow('Image3', mask)
cv2.waitKey(0)
cv2.destroyAllWindows