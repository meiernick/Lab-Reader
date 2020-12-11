#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2 as cv
import numpy as np


def getCenterPos(points):
    """
    Get the mean of all x and y Positions
    """

    # in case the input is empty
    if len(points) == 0:
        return (0, 0)

    # extract x and y positions
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    mean = np.array([sum(x) / len(points), sum(y) / len(points)])

    return mean


def angle(point):
    """
    Get the Angle of the Polar coordinates
    """

    angle = np.arctan2(point[1], point[0])

    return angle


def sortPointsClockwise(points):
    """
    Sort kartesian described Points by the Angle of the polar coordinates (clockwise)
    """
    # Got help from: https://www.baeldung.com/cs/sort-points-clockwise

    # Calculate the center point of all points
    mean = getCenterPos(points)

    # Subract the Center position from all points,
    # So they are located arround the point (0, 0)
    points = [(i[0]-mean[0], i[1]-mean[1]) for i in points]

    # Sort All points by angle
    points = sorted(points, key=angle)

    # Add the Center position from all points
    # Move them back from the center (0, 0) to where they were
    points = [(int(i[0]+mean[0]), int(i[1]+mean[1])) for i in points]

    return points


def drawSelection(img, points, index, active):
    """
    Visualize the  Selection with Lines and Points
    """

    # Nothing to draw if there are no points
    if len(points) == 0:
        return img

    alpha = 0.8
    overlay = img.copy()

    points = np.array(points)

    if active: # Everything in color
        # Draw red dots where the points are
        for i in range(len(points)):
            cv.circle(overlay, tuple(points[i]), 8, (0, 0, 255), -1, 16)
        # Draw the contour in green
        cv.drawContours(overlay, [points], 0, (0, 255, 0), 1, 16)
        # Draw the center of the x and y positions as a blue dot and if not active gray
        mean = [int(i) for i in getCenterPos(points)]
        radius = 10
        cv.circle(overlay, tuple(mean), 10, (255, 0, 0), -1, 16)
    else: # Everything in Gray
        # Draw red dots where the points are
        for i in range(len(points)):
            cv.circle(overlay, tuple(points[i]), 8, (127, 127, 127), -1, 16)
        # Draw the contour in green
        cv.drawContours(overlay, [points], 0, (127, 127, 127), 1, 16)
        # Draw the center of the x and y positions as a blue dot and if not active gray
        mean = [int(i) for i in getCenterPos(points)]
        radius = 10
        cv.circle(overlay, tuple(mean), 10, (127, 127, 127), -1, 16)


    cv.putText(overlay, str(index), tuple(
        (mean[0]-int(radius/2), mean[1]+int(radius/2))), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, 16)

    img = cv.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)
    return img


def pointDist(point1, point2):
    """
    Get the euclidean distance between two points
    """
    # x and y differences
    xdiff = abs(point1[0] - point2[0])
    ydiff = abs(point1[1] - point2[1])

    # Pythagoras
    distance = np.sqrt(xdiff**2 + ydiff**2)

    return distance


def scaleFrame(img, maxLen):
    """
    Resize an image, so it has a maximum length or width doesnt exeds the maxLen
    """

    y, x, c = img.shape

    scaleX = maxLen / x
    scaleY = maxLen / y

    scaleFactor = min(scaleX, scaleY)

    scaledImg = cv.resize(img, (int(x*scaleFactor), int(y*scaleFactor)))

    return scaledImg


def getCropEdges(width, height):
    """
    Get the edge Points of a rectangle with the desired width and height
    """

    pts = np.zeros((4, 2), dtype='float32')
    pts[0] = (0, 0)
    pts[1] = (width-1, 0)
    pts[2] = (width-1, height-1)
    pts[3] = (0, height-1)

    return pts


def getCropDim(rect):
    """
    Calculate the width and height of the selected rectangle

    To work porpperly the points have to be sorted:
    1. Upper Left
    2. Upper Right
    3. Bottom Right
    4. Bottom Left
    """

    (ul, ur, br, bl) = rect

    # TODO eventually Clean up:

    # # Take the average height and widht
    # width = ((ur[0] - ul[0]) + (br[0] - bl[0])) / 2
    # height = ((br[1] - ur[1]) + (br[1] - ul[1])) / 2

    # # Take the max height and widht
    # width = max([ur[0]-ul[0], br[0]-bl[0]])
    # height = max([br[1]-ur[1], bl[1]-ul[1]])

    # Take the max height and widht (with pytagoras)
    # width = max([np.sqrt((ur[0]-ul[0])**2 + (ur[1]-ul[1])**2), np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)])
    # height = max([np.sqrt((br[1]-ur[1])**2 + (br[0]-ur[0])**2), np.sqrt((bl[1]-ul[1])**2 + (bl[0]-ul[0])**2)])

    # Take the avg height and widht (with pytagoras)
    upperWidth = np.sqrt((ur[0]-ul[0])**2 + (ur[1]-ul[1])**2)
    bottomWidth = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    rightHeight = np.sqrt((br[1]-ur[1])**2 + (br[0]-ur[0])**2)
    leftHeight = np.sqrt((bl[1]-ul[1])**2 + (bl[0]-ul[0])**2)
    width = (upperWidth + bottomWidth) / 2
    height = (rightHeight + leftHeight) / 2

    return width, height


# Global variables related to onMouse()
ptsClick = [[] for i in range(10)]      # Here the selected points get saved in 10 sets
activePointSet = 1      # Tells which Set is active
lButtonDown = False     # Stores the Left-Buttons last state
activeDrag = None       # None=No_Dragging, >0=Index_of_dragged_Point, <0=All_dragged


def onMouse(event, x, y, flags, param):
    """
    Handle the Mouse

    Right Click: Clears the stored Points
    Left Click: Either creates a new Point or Drags an existing one

    Note! This function uses 3 global variables!
    """

    global ptsClick     # Here the selected points get saved
    global activePointSet  # Tells which Set is active
    global lButtonDown  # Stores the Left-Buttons last state
    global activeDrag   # 0=No_Dragging, >0=Index_of_dragged_Point

    # Right Click
    if event == cv.EVENT_RBUTTONUP:
        ptsClick[activePointSet].clear()

    # Left Down
    if event == cv.EVENT_LBUTTONDOWN:
        lButtonDown = True

    # Left Click
    if event == cv.EVENT_LBUTTONUP:
        lButtonDown = False
        activeDrag = None
        # Only allow 4 Points
        if len(ptsClick[activePointSet]) < 4:
            ptsClick[activePointSet].append((x, y))

    # Draging
    if event == cv.EVENT_MOUSEMOVE and lButtonDown:
        if activeDrag != None:
            if activeDrag < 0:
                # Drag all points togeter
                center = [int(i)
                          for i in getCenterPos(ptsClick[activePointSet])]
                x_diff, y_diff = x-center[0], y-center[1]
                ptsClick[activePointSet] = [[i[0]+x_diff, i[1]+y_diff]
                                            for i in ptsClick[activePointSet]]

            else:
                # Drag the current dragged point
                ptsClick[activePointSet][activeDrag] = (x, y)
        else:
            dragRadius = 30  # How many pixels apart a point is allowed to be, to catch on

            if pointDist((x, y), getCenterPos(ptsClick[activePointSet])) < dragRadius:
                activeDrag = -1
                # Drag all points togeter
                center = [int(i)
                          for i in getCenterPos(ptsClick[activePointSet])]
                x_diff, y_diff = x-center[0], y-center[1]
                ptsClick[activePointSet] = [[i[0]+x_diff, i[1]+y_diff]
                                            for i in ptsClick[activePointSet]]

            # Check if a point to drag is in reach
            for i in range(len(ptsClick[activePointSet])):
                if pointDist((x, y), ptsClick[activePointSet][i]) < dragRadius:
                    ptsClick[activePointSet][i] = (x, y)
                    activeDrag = i
                    # make shure only one point gets dragged
                    break


def getWarp(img, index):
    """
    Warp the passed image according to the selected points

    Note! This function uses a global variable!
    """

    global ptsClick     # Here the selected points get saved
    # global activePointSet  # Tells which Set is active

    ptsSorted = sortPointsClockwise(ptsClick[index])

    # Warp is only possible if 4 edges are selected
    if len(ptsClick[index]) >= 4:
        width, height = getCropDim(ptsSorted)
        ptsCrop = getCropEdges(width, height)

        rect = np.zeros((4, 2), dtype='float32')
        for i in range(len(rect)):
            rect[i][0] = ptsSorted[i][0]
            rect[i][1] = ptsSorted[i][1]

        M = cv.getPerspectiveTransform(rect, ptsCrop)
        warp = cv.warpPerspective(img, M, (int(width), int(height)))

        return warp

    else:
        return None


def getDraw(img):
    """
    Draw the Selection onto the image

    Note! This function uses a global variable!
    """

    global ptsClick     # Here the selected points get saved
    global activePointSet  # Tells which Set is active

    img_draw = img.copy()
    for i in range(len(ptsClick)):
        ptsSorted = sortPointsClockwise(ptsClick[i])
        img_draw = drawSelection(img_draw, ptsSorted, i, i==activePointSet)

    return img_draw


def setActivePointSet(index):
    """
    Set the active point set, to switch between selections
    """
    global activePointSet  # Tells which Set is active
    activePointSet = int(index)


def main(argv):
    """ Main program """
    pass


if __name__ == '__main__':
    sys.exit(main(sys.argv))
