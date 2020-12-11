#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import sys

def windowRect(searchName):
    """
    Searches the Window with the title, and returns som stuff

    # Parameters:
        searchName  The name to search for in the title
        (not case sensitive)

    # Returns: a namedtuple
        .name   title of the window
        .x      x-position of origin
        .y      y-position of origin
        .w      width
        .h      height

    # Example:
        window = windowRect('editor')
        print('x-Position:', window.x)
        print('y-Position:', window.y)
    """
    import win32gui
    from collections import namedtuple
    foundWindows = [] # Save all found windows
    window = namedtuple('window', 'name x y w h')
    def callback(hwnd, extra):
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y
        name = win32gui.GetWindowText(hwnd)
        # Only append if the window contains the name
        if searchName.lower() in name.lower():
            foundWindows.append(window(name, x, y, w, h))

    # Get all Window Names
    win32gui.EnumWindows(callback, None)

    return foundWindows

# windowRect('editor')


def main(argv):
    """ Main program """
    import numpy as np
    import cv2

    size = 200
    img1 = np.ones((size,size,3))*(0,255,0)
    img2 = np.ones((size,size,3))*(0,0,255)

    while(True):

        # Show both windows
        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        # Get the window Position of the first Window
        w = windowRect('img1')[0]
        # Place the second window right beside the first one
        cv2.moveWindow('img2', w.x+w.w-15, w.y)


        c = cv2.waitKey(1)
        if c == 0x1B:
            break  # close on ESC key

    cv2.destroyAllWindows()



if __name__ == '__main__':
    sys.exit(main(sys.argv))
