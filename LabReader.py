# %% Import Libraries
import sys
import cv2 as cv
import numpy as np

import csv_stuff as lrc
import warp_stuff as lrw
import windowRect as wr
import number_reader

# Setup
filename = 'values.csv'  # Name of the CSV-File, where the values get stored
mainWindowName = 'Select Display (close with ESC/ Q)'
warpWindowName = 'Selcted Display (close with ESC/ Q)'
nr = number_reader.number_reader()
cv.namedWindow(mainWindowName)
cv.setMouseCallback(mainWindowName, lrw.onMouse)

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture(r'../Beispielmaterial/2020-10-12 12.49.37.mp4')
# cap = cv.VideoCapture(r'../Beispielmaterial/2020-10-12 12.50.06.mp4')
# cap = cv.VideoCapture(r'../Beispielmaterial/2020-10-12 12.51.40.mp4')
# cap = cv.VideoCapture(r'../Beispielmaterial/2020-10-12 12.52.14.mp4')
# cap = cv.VideoCapture(r'../Beispielmaterial/2020-10-12 12.52.41.mp4')
# cap = cv.VideoCapture(r'../Beispielmaterial/2020-10-12 14.15.33.mp4')
# cap = cv.VideoCapture(r'../Beispielmaterial/2020-10-12 14.15.54.mp4')
# cap = cv.VideoCapture(r'../Beispielmaterial/20201109_231608.mp4')

print('Select the 4 edges of the Display you want to read')
print('-\tyou can also drag the points if you like to adjust them')
print('-\tright-click resets all points')
print('')
print('Exit with:                     ESC or Q')
print('Read the Display with:         ENTER')
print('Delete the last saved Value:   BACKSPACE')
print('')
print('')
print('have Fun :)')
print('')
print('')


# Loop
while(True):
    ret, frame = cap.read()

    # TODO Remove that part, it is only used to rewind the Video :)
    if ret == False:
        # Jump to pos 0 to restart the video
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    # prevent the window from being too large for the screen
    frame = lrw.scaleFrame(frame, 800)

    # Display the resulting frame
    cv.imshow(mainWindowName, lrw.getDraw(frame))

    # Get the postition of the main window
    mainWindowRect = wr.windowRect(mainWindowName)[0]
    warp_x, warp_y = mainWindowRect.x + mainWindowRect.w - 10, mainWindowRect.y
    detectedValues = [[] for i in range(10)]
    for i in range(10):
        warp = lrw.getWarp(frame, i)
        if not warp is None:
            detectedValue, warp = nr.read_number_from_img(warp)
            if detectedValue != None:
                if (len(detectedValue) == 1):
                    detectedValues[i] = detectedValue[0]
                else:
                    detectedValues[i] = detectedValue

            cv.imshow(str(i)+' '+warpWindowName, warp)
            cv.moveWindow(str(i)+' '+warpWindowName, warp_x, warp_y)
            warp_y += np.shape(warp)[0] + 35  # 35 Because the Titlebar
        else:
            cv.destroyWindow(str(i)+' '+warpWindowName)

    c = cv.waitKey(1) & 0xFF
    if c == ord('q'):
        break  # close on Key Q
    if c == 0x1B:
        break  # close on ESC key
    if c == 0x0D:
        # Save Value on Enter Key
        lrc.saveValue(detectedValues, filename=filename)
    if c == ord('\b'):
        lrc.removeValue(filename=filename)  # Delete Value on BACKSPACE Key
    if chr(c).isdigit() == True:
        lrw.setActivePointSet(int(chr(c)))

# Clean Up
cap.release()
cv.destroyAllWindows()

# %%
