
import pyautogui
import numpy as np
from scipy.interpolate import BPoly
import cv2
import mss
import random
import time

pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# lets do a woodcutting bot, the should be able to cut multiple trees and drop inventory then restart.


def getObjectOutlines(frame: cv2.Mat) -> list:
    """Find the contours of the red shapes and return their positions.

    Args:
        frame (cv2.Mat): image

    Returns:
        list: contour of the object
    """

    # Filter color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 200, 200])
    upper_red = np.array([10, 255, 255])

    # Finding the Mask
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Finding the contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [x for x in contours if x.size > 50]

    return contours


def checkIfChopping(frame, box):
    sub_frame = frame[box['left']:box['left'] + box['height'], box['top']:box['top'] + box['width']]

    # Filter color
    hsv = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Finding the Mask
    mask = cv2.inRange(hsv, lower_red, upper_red)

    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 5:
        return False

    return True


def checkIfInventoryFull(frame, box):
    sub_frame = frame[box['left']:box['left'] + box['height'], box['top']:box['top'] + box['width']]

    gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    dilated = cv2.dilate(canny, (1, 1), iterations=3)

    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [x for x in contours if x.size > 50]
    print(len(contours))

    if len(contours) >= 28:
        new_contours = []
        for cnt in contours:
            cnt = cnt + (box['top'], box['left'])
            new_contours.append(cnt)

        return True, new_contours
    return False, contours


def getRandomClickCoordinates(shape: list) -> list:
    """Find a random coordinates inside the shape. The shape is assume to be always convex.

    Args:
        shape (list): list of points representing the point

    Returns:
        list: random coordinate inside the shape
    """

    # Find min/max x and y
    min_x = np.min(shape[:, 0][:, 0])
    max_x = np.max(shape[:, 0][:, 0])
    min_y = np.min(shape[:, 0][:, 1])
    max_y = np.max(shape[:, 0][:, 1])

    # Find a point inside the polygon
    L = len(shape)
    point = None

    count = 0
    while count % 2 == 0:
        count = 0
        # Initial guess
        point = [random.randrange(min_x, max_x + 1), random.randrange(min_y, max_y + 1)]
        i = 0

        # Still have a small bug when the line cross a vertex
        while True:
            c = point.copy()
            p1 = shape[:, 0][i, :]
            p2 = shape[:, 0][(i + 1) % L, :]

            isCrossing = False

            order_y = sorted([p1[1], p2[1]])
            if order_y[0] < c[1] < order_y[1]:

                order_x = sorted([p1[0], p2[0]])
                if order_x[0] <= c[0] <= order_x[1]:
                    if c[0] >= order_x[0]:
                        if p1[0] == p2[0]:
                            x = p1[0]
                        else:
                            m = (p1[1] - p2[1]) / (p1[0] - p2[0])
                            b = p1[1] - m * p1[0]
                            x = (c[1] - b) / m

                        if x <= c[0]:
                            isCrossing = True
                elif c[0] < order_x[0]:
                    isCrossing = True

            if isCrossing:
                count = count + 1

            i = (i + 1) % L
            if i == 0:
                break

    return point


def moveMouseHumanWay(coordinate, min_time, max_time, num_points):

    start = pyautogui.position()

    cp_num = random.randint(4, 8)
    cp = np.zeros((cp_num, 2))

    cp[0, :] = start
    cp[cp_num-1, :] = coordinate

    # Assign control points
    order_x = sorted([start[0], coordinate[0]])
    order_y = sorted([start[1], coordinate[1]])
    for i in range(1, cp_num-1):
        cp[i, :] = [random.randint(*order_x), random.randint(*order_y)]

    sorted_cp = cp[np.argsort(cp[:, 0])]

    curve = BPoly(sorted_cp[:, None, :], [0, 1])
    x = np.linspace(0, 1, num_points)
    points = np.round_(curve(x))

    if coordinate[0] < start[0]:
        points = points[np.argsort(points[:, 0])][::-1]

    duration = random.uniform(min_time, max_time)
    timeout = duration / len(points)
    for p in points:
        pyautogui.moveTo(p[0], p[1])
        time.sleep(timeout)

    pyautogui.leftClick()


def main():
    while True:
        time.sleep(random.uniform(2, 3.5))
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {"top": 0, "left": 0, "width": 960, "height": 665}
            # Grab the data
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)

        # absolute_path = os.path.dirname(__file__)
        # fileName = 'not_woodcutting.png'
        # path = os.path.join(absolute_path, fileName)
        # img = cv2.imread(path)

        # Check if inventory is full
        box = {"top": 705, "left": 300, "width": 210, "height": 315}
        isFull, inventoryContours = checkIfInventoryFull(img, box)

        if isFull:
            print('Start Dropping items')
            pyautogui.keyDown('shift')
            for obj in inventoryContours:
                # Find next click
                coord = getRandomClickCoordinates(obj)
                moveMouseHumanWay(coord, 0.2, 0.7, 10)
            pyautogui.keyUp('shift')


        # Find if player is idle or not
        box = {"top": 10, "left": 50, "width": 160, "height": 80}
        isChopping = checkIfChopping(img, box)

        if not isChopping and not isFull:
            # Filter color
            treesContours = getObjectOutlines(img)

            if len(treesContours) != 0:
                # Select random shape
                index = random.randrange(0, len(treesContours))

                # Find next click
                coord = getRandomClickCoordinates(treesContours[index])

                moveMouseHumanWay(coord, 0.5, 1.5, 30)

                # img = cv2.circle(img, coord, radius=1, color=(0, 255, 255), thickness=1)
    # cv2.imshow('img', img)
    #
    # if cv2.waitKey(0):
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
