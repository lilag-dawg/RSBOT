import pyautogui
import numpy as np
import cv2
import mss
import random
import time


# lets do a woodcutting bot, the should be able to cut multiple trees and drop inventory then restart.


# Find the contours of the red shapes and return their positions
def getContoursOutlines(frame):

    # Filter color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,255,200])
    upper_red = np.array([0,255,255])

    # Finding the Mask
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Finding the contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    contours = [x for x in contours if x.size > 50]

    return contours

# Find a random coordinates inside the shape. The shape is assume to be always convex.
def getRandomClickCoordinates(shape):

    # Select random index
    index = random.randrange(0,len(shape))

    # Find min/max x and y
    min_x = np.min(shape[index][:,0][:,0])
    max_x = np.max(shape[index][:,0][:,0])
    min_y = np.min(shape[index][:,0][:,1])
    max_y = np.max(shape[index][:,0][:,1])

    L = len(shape[index])

    point = None
    count = 0
    while count % 2 == 0:
        count = 0
        # Initial guess
        point = [random.randrange(min_x,max_x + 1), random.randrange(min_y,max_y + 1)]

        # Check if point is inside polygon
        # todo: if P.y = V1.y or V2.y for two consecutive segment, algorithm doesnt treat it
        i = 0
        for v1 in shape[index][:,0]:
            if i+1 >= L:
                break
            v2 = shape[index][:,0][i+1,:]
            isCrossing = False

            order_y = sorted([v1[1],v2[1]])

            if order_y[0] <= point[1] <= order_y[1]:
                order_x = sorted([v1[0], v2[0]])
                if order_x[0] <= point[0] <= order_x[1]:
                    if v1[1] == v2[1] == point[1] or v1[1] == v2[1]:
                        isCrossing = True
                    if v1[0] != v2[0]:
                        m = (v1[1] - v2[1]) / (v1[0] - v2[0])
                        b = v1[1] - m * v1[0]
                        x = (point[1] - b) / m
                    else:
                        x = v1[0]

                    if x > point[0]:
                        isCrossing = True

                elif order_x[0] > point[0]:
                    isCrossing = True

            if isCrossing:
                count = count + 1
            i = i + 1

    return point





def main():
    # with mss.mss() as sct:
    #     # The screen part to capture
    #     monitor = {"top": 187, "left": 5, "width": 865, "height": 788}
    #     # Grab the data
    #     sct_img = sct.grab(monitor)
    #     img = np.array(sct_img)

    # path
    path = r'C:\Users\alexi\Documents\PythonProject\pythonProject\RSBOT\marker_screenshot.png'
    img = cv2.imread(path)

    # Filter color
    contours = getContoursOutlines(img)

    # Find next click
    coord = getRandomClickCoordinates(contours)
    img = cv2.circle(img, coord, radius=1, color=(0,255,255), thickness=2)

    cv2.imshow('img', img)

    if cv2.waitKey(0):
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

