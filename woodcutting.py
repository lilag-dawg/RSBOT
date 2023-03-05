import math
import random
import secrets

import cv2
import mss
import pyautogui
import requests
import time
import numpy as np

import mouse


def req_get(endpoint):
    response = {}
    try:
        response = requests.get(f"http://localhost:8081/{endpoint}", timeout=1)
    except ConnectionError as e:
        print(e)

    return response.json()


def check_if_idle(endpoint):
    data = req_get(endpoint)
    if data.get("animation") != -1 or data.get("animation pose") not in [808, 813]:
        return False
    return True


def check_inv(endpoint):
    data = req_get(endpoint)
    i = 0
    for item in data:
        if item['id'] != -1:
            i = i + 1
    if i == 28:
        return True, data
    return False, data


def findInventorySlothSize(box, edge):
    # Find Rectangles
    inv_info = []  # Inventory is ordered row by row starting with top left cell, row 0 to 6

    x_div = math.floor(box['width'] / 4)
    y_div = math.floor(box['height'] / 7)

    for j in range(7):
        for i in range(4):
            border = np.ndarray(shape=(4, 1, 2), dtype=int)
            border[0] = [i * x_div + edge + box['top'], j * y_div + edge + box['left']]
            border[1] = [i * x_div + edge + box['top'], (j + 1) * y_div - edge + box['left']]
            border[2] = [(i + 1) * x_div - edge + box['top'], j * y_div + edge + box['left']]
            border[3] = [(i + 1) * x_div - edge + box['top'], (j + 1) * y_div - edge + box['left']]

            inv_info.append(border)

    return inv_info


def assign_id_to_sloths(data, contours_list):
    sloths = [[-1, None] for i in range(28)]

    k = 0
    for item in data:
        sloths[k] = [item['id'], contours_list[k]]
        k = k + 1

    return sloths


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


def pnpoly(contour):
    # Find min/max x and y
    min_x = np.min(contour[:, 0][:, 0])
    max_x = np.max(contour[:, 0][:, 0])
    min_y = np.min(contour[:, 0][:, 1])
    max_y = np.max(contour[:, 0][:, 1])

    # Find a point inside the polygon
    L = len(contour)
    point = None

    test = -1
    while test <= -1:
        # Guess
        point = [boundedGaussian(min_x, max_x + 1), boundedGaussian(min_y, max_y + 1)]

        test = cv2.pointPolygonTest(contour, point, False)
    return point


def boundedGaussian(low, high):
    mean = low + (high - low) / 2
    std = (high - low) / 5

    return round(random.gauss(mean, std))


def findClosestObjectFromPlayer(contours):
    # search the object closest to the mouse

    L = len(contours)
    center = np.zeros((L, 2), dtype=int)

    k = 0
    for c in contours:
        # order is minX, maxX, minY, maxY
        min_x = np.min(c[:, 0][:, 0])
        max_x = np.max(c[:, 0][:, 0])
        min_y = np.min(c[:, 0][:, 1])
        max_y = np.max(c[:, 0][:, 1])

        center[k, :] = [min_x + round((max_x - min_x) / 2), min_y + round((max_y - min_y) / 2)]
        k = k + 1

    cursor = pyautogui.position()

    # find closest center
    distance = np.zeros((L, 1), dtype=int)
    for i in range(L):
        distance[i] = math.sqrt(pow(cursor[0] - center[i, 0], 2) + pow(cursor[1] - center[i, 1], 2))

    return contours[np.argmin(distance)]


def dropInventory(sloths, keep, dropOrder):
    # reorder the list
    sloths = [sloths[i] for i in dropOrder]

    pyautogui.keyDown('shift')
    for s in sloths:
        if keep.count(s[0]) <= 0 and s[0] != -1:
            p = pnpoly(s[1])
            mouse.move(p, 14, 2, [0.3, 0.8])
            mouse.real_click()

    pyautogui.keyUp('shift')


def random_chance(probability: float) -> bool:
    """
    Returns true or false based on a probability.
    Args:
        probability: The probability of returning true (between 0 and 1).
    Returns:
        True or false.
    """
    # ensure probability is between 0 and 1
    if not isinstance(probability, float):
        raise TypeError("Probability must be a float")
    if probability < 0.000 or probability > 1.000:
        raise ValueError("Probability must be between 0 and 1")
    return secrets.SystemRandom().random() < probability


def logout():
    door_coords = [792, 625, 820, 660]  # min_x min_y max_x, max_y
    logout_coords = [725, 560, 890, 590]  # min_x min_y max_x, max_y

    door_border = np.ndarray(shape=(4, 1, 2), dtype=int)
    button_border = np.ndarray(shape=(4, 1, 2), dtype=int)

    door_border[0] = [door_coords[0], door_coords[1]]
    door_border[1] = [door_coords[0], door_coords[3]]
    door_border[2] = [door_coords[2], door_coords[1]]
    door_border[3] = [door_coords[2], door_coords[3]]

    button_border[0] = [logout_coords[0], logout_coords[1]]
    button_border[1] = [logout_coords[0], logout_coords[3]]
    button_border[2] = [logout_coords[2], logout_coords[1]]
    button_border[3] = [logout_coords[2], logout_coords[3]]

    # click on door
    p1 = pnpoly(door_border)
    mouse.move(p1, 4, 4, [0.3, 0.8])
    mouse.real_click()

    # click on logout
    p2 = pnpoly(button_border)
    mouse.move(p2, 2, 2, [0.3, 0.8])
    mouse.real_click()


def main():
    # ------------ Modify depending on use --------------- #

    keepers = [1359, 2121]  # items id to keep in inventory
    minutes = 30  # minutes until logout, maximum is 360 minutes (6 hours)
    run_time = minutes * 60
    # ------------ End of modifications --------------- #

    # The screen part to capture
    monitor_dim = {"top": 0, "left": 0, "width": 960, "height": 665}

    # The screen part of the in-game inventory
    inv_dim = {"top": 705, "left": 300, "width": 200, "height": 310}

    inv_sloth_contours = findInventorySlothSize(inv_dim, 6)

    startTime = time.time()

    while True:
        time.sleep(random.uniform(0.75, 1.5))

        isInventoryFull, item_ids = check_inv('inv')
        isPlayerIdle = check_if_idle('events')

        sloths = assign_id_to_sloths(item_ids, inv_sloth_contours)

        # take random breaks
        if random_chance(probability=0.005):
            print('taking a lil break')
            time.sleep(random.uniform(55, 65))

        # move mouse
        if random_chance(probability=0.15):
            print('move mouse a bit')
            mouse.idle()

        # drop inventory
        if isInventoryFull or random_chance(probability=0.01):
            print('Dropping inventory')
            order = list(range(28))  # todo: change order patterns randomly
            dropInventory(sloths, keepers, order)

        with mss.mss() as sct:
            # Grab the data
            sct_img = sct.grab(monitor_dim)
            img = np.array(sct_img)

            treesContours = getObjectOutlines(img)

            if time.time() - startTime >= run_time:
                logout()
                if len(treesContours) == 0:
                    exit()

            if len(treesContours) != 0:
                obj = findClosestObjectFromPlayer(treesContours)

                if isPlayerIdle:
                    print('click on tree')
                    p = pnpoly(obj)
                    mouse.move(p, 12, 2, [0.5, 1.5])
                    mouse.real_click()


if __name__ == "__main__":
    main()
