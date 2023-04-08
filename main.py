import cv2
import numpy as np
import matplotlib.pyplot as plot

image = cv2.imread('./imgs/image1.jpg')
height, width = 560, 280

def draw_rectangles(ctrs, img):
    
    output = img.copy()
    
    for i in range(len(ctrs)):
    
        M = cv2.moments(ctrs[i]) # moments
        rot_rect = cv2.minAreaRect(ctrs[i])
        w = rot_rect[1][0] # width
        h = rot_rect[1][1] # height
        
        box = np.int64(cv2.boxPoints(rot_rect))
        cv2.drawContours(output,[box],0,(0,0,255),2) # draws box
        
    return output

def filter_contours(ctrs, min_s = 40, max_s = 600, alpha = 8):  
    
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

def find_contours_color(ctrs, input_img):

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

def draw_holes(input_img, color3 = (200,140,0)):
        
    color = (190, 190, 190) # gray color
    color2 = (120, 120, 120) #  gray color, for circles (holes) on generated img

    img = input_img.copy() # make a copy of input image
    
    # borders 
    cv2.line(img,(0,0),(width,0),color3,3) # top
    cv2.line(img,(0,height),(width,height),color3,3) # bot
    cv2.line(img,(0,0),(0,height),color3,3) # left
    cv2.line(img,(width,0),(width,height),color3,3) # right
    
    # adding circles to represent holes on table
    cv2.circle(img, (0, 0), 11,color, -1) # top right
    cv2.circle(img, (width,0), 11, color, -1) # top left
    cv2.circle(img, (0,height), 11, color, -1) # bot left
    cv2.circle(img, (width,height), 11, color, -1) # bot right
    cv2.circle(img, (width,int(height/2)), 8, color, -1) # mid right
    cv2.circle(img, (0,int(height/2)), 8, color, -1) # mid left
    
    # adding another, smaller circles to the previous ones
    cv2.circle(img, (0, 0), 9,color2, -1) # top right
    cv2.circle(img, (width,0), 9, color2, -1) # top left
    cv2.circle(img, (0,height), 9, color2, -1) # bot left
    cv2.circle(img, (width,height), 9, color2, -1) # bot right
    cv2.circle(img, (width,int(height/2)), 6, color2, -1) # mid right
    cv2.circle(img, (0,int(height/2)), 6, color2, -1) # mid left
    
    return img

def create_table():
    
    # new generated img 
    img = np.zeros((height,width,3), dtype=np.uint8) # create 2D table image 
    img[:, :] = [200, 200, 200] # setting RGB colors to green pool table color, (0,180,10)=certain green
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    
    # create circle in the right size
    cv2.circle(img, (int(width/2),int(height/5)), # center of circle
               int((width/3)/2), # radius
               (100,100,100)) # color
    
    # delete half of circle by coloring in green color
    img[int(height/5):height,0:width] = [200, 200, 200] 
    # create line
    cv2.line(img,(0,int(height/5)),(width,int(height/5)),(100,100,100)) 
    
    return img

def draw_table(ctrs,background = create_table(), radius=7, size = -1, img = 0):

    K = np.ones((3,3),np.uint8) # filter
    
    final = background.copy() # canvas
    mask = np.zeros((560, 280),np.uint8) # empty image, same size as 2d generated final output
    
    
    for x in range(len(ctrs)): # for all contours
        
        # find center of contour
        M = cv2.moments(ctrs[x])
        cX = int(M['m10']/M['m00']) # X pos of contour center
        cY = int(M['m01']/M['m00']) # Y pos
        
        # find color average inside contour
        mask[...]=0 # reset the mask for every ball 
        cv2.drawContours(mask,ctrs,x,255,-1) # draws mask for each contour
        mask =  cv2.erode(mask,K,iterations = 3) # erode mask several times to filter green color around balls contours
        
        
        # balls design:
        
        
        # circle to represent snooker ball
        final = cv2.circle(final, # img to draw on
                           (cX,cY), # position on img
                           radius, # radius of circle - size of drawn snooker ball
                           cv2.mean(img,mask), # color mean of each contour-color of each ball (src_img=transformed img)
                           size) # -1 to fill ball with color
        
        # add black color around the drawn ball (for cosmetics)
        final = cv2.circle(final, (cX,cY), radius, 0, 0) 
        
        # small circle for light reflection
        final = cv2.circle(final, (cX-2,cY-2), 0, (255,255,255), -1)
         
        

        
    return final

coordsImage = {
    'tl': (164, 107),
    'tla': [164, 107],
    'bl': (164, 337),
    'bla': (164, 337),
    'tr': (636, 102),
    'tra': (636, 102),
    'br': (639, 337),
    'bra': (639, 337)
}

# coordsDiagram = {
#     'tl': (0, 0),
#     'tla': [0, 0],
#     'bl': (0, width),
#     'bla': [0, width],
#     'tr': (width, height),
#     'tra': [width, height],
#     'br': (height, width),
#     'bra': [height, width]
# }

coordsDiagram = {
    'tl': (0, 0),
    'tla': [0, 0],
    'bl': (0, width),
    'bla': [0, width],
    'tr': (height, 0),
    'tra': [height, 0],
    'br': (height, width),
    'bra': [height, width]
}

table = image.copy()
cv2.circle(table, coordsImage['tl'], 8, (0, 0, 255), -1) # top left pocket
cv2.circle(table, coordsImage['bl'], 8, (0, 0, 255), -1) # bot left pocket
cv2.circle(table, coordsImage['tr'], 8, (0, 0, 255), -1) # top right pocket
cv2.circle(table, coordsImage['br'], 8, (0, 0, 255), -1) # bot right pocket

newImage = np.zeros((width, height, 3), dtype=np.uint8)
finalImage = newImage.copy()
cv2.circle(finalImage, coordsDiagram['tl'], 8, (0, 0, 255), -1) # top left pocket
cv2.circle(finalImage, coordsDiagram['bl'], 8, (0, 0, 255), -1) # bot left pocket
cv2.circle(finalImage, coordsDiagram['br'], 8, (0, 0, 255), -1) # bot right pocket
cv2.circle(finalImage, coordsDiagram['tr'], 8, (0, 0, 255), -1) #  top right pocket

firstPoints = np.float32([coordsImage['tla'], coordsImage['bla'], coordsImage['tra'], coordsImage['bra']])
secondPoints = np.float32([[0,height],[width,height],[0,0],[width,0]])
matrix = cv2.getPerspectiveTransform(firstPoints, secondPoints)
warped = cv2.warpPerspective(image, matrix, (width, height))


hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
kernel = np.ones((5,5), np.uint8)

lower_bound = np.array([100, 230, 230])
upper_bound = np.array([110, 255, 255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)
mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
_,mask_invert = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)
masked = cv2.bitwise_and(warped, warped, mask=mask_invert)
# mask_others = cv2.bitwise_not(mask)
# maskInvert = cv2.bitwise_and(warped, warped, mask=mask_others)

contours, hierarchy = cv2.findContours(mask_invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
detected_objects = draw_rectangles(contours, warped)
contours_final = filter_contours(contours)
detected_objects_final = draw_rectangles(contours_final, warped)

contours_color = find_contours_color(contours_final, warped)
contours_color = cv2.addWeighted(contours_color, 0.5, warped, 0.5, 0)

diagram = draw_table(contours_final, img=warped)
diagram = draw_holes(diagram)

cv2.imshow('table', table)
cv2.imshow('test', finalImage)
cv2.imshow('obj', detected_objects)
cv2.imshow('filtered obj', detected_objects_final)
cv2.imshow('ball colors', contours_color)
cv2.imshow('original', image)
cv2.imshow('diagram', diagram)
cv2.waitKey(0)
cv2.destroyAllWindows