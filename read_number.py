import numpy as np
import cv2 as cv
from cv2 import ml
from matplotlib import pyplot as plt
from time import process_time
from collections import namedtuple

position = namedtuple('position', 'x y w h')
symbol_data = namedtuple('symbol_data', 'data pos')


class number_reader:

    _winSize = (20, 20)
    _cellSize = (8, 8)
    _blockSize = (8, 8)
    _blockStride = (4, 4)
    _nbins = 9
    _signedGradients = True
    _derivAperture = 1
    _winSigma = -1.
    _histogramNormType = 0
    _L2HysThreshold = 0.2
    _gammaCorrection = 0
    _nlevels = 64

    def __init__(self):
        self._hog = cv.HOGDescriptor(
            self._winSize,
            self._blockSize,
            self._blockStride,
            self._cellSize,
            self._nbins,
            self._derivAperture,
            self._winSigma,
            self._histogramNormType,
            self._L2HysThreshold,
            self._gammaCorrection,
            self._nlevels,
            self._signedGradients
        )

        self._svm = ml.SVM_load("svm_data.yaml")

        self._kernel = np.matrix([[0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0]], np.uint8)

    def _calculate_distances(self, symbols, median_ypos, median_width):
        """
            calculate the distances between a symbol and the folowing one for every symbol
        """
        distances = []
        for i in range(len(symbols) - 1):
            # end of the leading symbol: x-pos + width
            end_leading = symbols[i].pos.x + symbols[i].pos.w
            # beginning of the trailing symbol: x-pos
            start_trailing = symbols[i + 1].pos.x
            if symbols[i + 1].data == "1":
                # if the digit is an one add the difference in width to the digit to compensate the smaller with of the one.
                # this is necessary to make sure the one is not recognized as a space
                start_trailing = start_trailing -\
                    median_width + symbols[i + 1].pos.w
            distance = start_trailing - end_leading
            # append a tuple containing x, y, w, h of the space
            distances.append(symbol_data(
                distance, position(end_leading, median_ypos, distance, 1)))
        return distances

    def read_number_from_img(self, img):
        """
            Read a number from an image of a 7-Segment
            All digits need to be in one line
            Multiple numbers can be separated if their distance is bigger than the width of one digit
            No digits can be detected if they touch the border
        """
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

        eroded = cv.erode(thresh, self._kernel)

        # ## Detect points
        # Detect decimal points from the aspect ratio. Due to the eroding before the points are now not round anymore but rather flat.
        # The numbers stay far higher than they are width, thus if a shape is wider than high it cant be a number. Their position is saved and they are deleted from the threshold image
        contours, hierarchies = cv.findContours(
            eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        points = []

        for (cnt, hierarchy) in zip(contours, hierarchies[0]):
            if hierarchy[3] < 0:
                x, y, w, h = cv.boundingRect(cnt)
                area = cv.contourArea(cnt)
                perimeter = cv.arcLength(cnt, True)
                S = 4 * np.pi * area / perimeter**2
                if w > h and x != 0 and y != 0:
                    points.append(position(x, y, w, h))
                    cv.rectangle(thresh, (x, y - 5),
                                 (x+w, y+h + 5), (0, 0, 0), -1)

        blurred = cv.medianBlur(thresh, 3)
        number_of_components, labels = cv.connectedComponents(blurred)

        component_imgs = []
        for component in range(1, number_of_components):
            componentImg = labels == component
            rect = cv.boundingRect(componentImg.astype(np.uint8))
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]

            # only cut out those shape that do not touch the edge of the image
            # as numbers should be in the middle of the image
            if x != 0 and y != 0 and x + w != thresh.shape[1] and y + h != thresh.shape[0]:
                cut = (componentImg[y - 1:y + h + 1,
                                    x - 1:x + w + 1]).astype(np.uint8)

                component_imgs.append(symbol_data(cut, position(x, y, w, h)))

                cv.rectangle(img, (x - 1, y - 1),
                             (x+w + 1, y+h + 1), (0, 255, 0), 2)

        # resize every valid symbol detected before to the required size for the hog-calculation
        cut_components = [symbol_data(cv.resize(component.data, (20, 20)), component.pos)
                          for component in component_imgs]

        # calculate the hog-feature-vector for every extracted and resized symbol
        hogs = [self._hog.compute(component.data)
                for component in cut_components]

        # let the svm predict a value for every detected number
        data = np.float32(hogs)
        result = self._svm.predict(data)

        # create a list of tuples containing the position of a symbol and its detected value
        symbols = [symbol_data(str(int(res)), cut.pos)
                   for (cut, res) in zip(cut_components, result[1])]

        # calculate the median heigth, width and y-position of every detected symbol.
        # these values are used to determine which detectet points are decimal points.
        # also they are used to determine where one number ends and a new one begins.

        median_ypos = np.median([symbol.pos.y for symbol in symbols])

        median_height = np.median([symbol.pos.h for symbol in symbols])

        median_width = np.median([symbol.pos.w for symbol in symbols])

        # Check every previously detected point. if it is near enough to the numbers it is a decimal point
        for point in points:
            # if the point is within half a number height around the bottom of the numbers it is a decimal point
            if (point.y > (median_ypos + 0.75 * median_height)) and (point.y < (median_ypos + 1.25 * median_height)):
                symbols.append(symbol_data('.', point))

        # sort the list by the x position
        symbols = list(sorted(symbols, key=lambda sym: sym.pos.x))

        distances = self._calculate_distances(
            symbols, median_ypos, median_width)

        # Add a space to the symbol list if the distance between two numbers is bigger than the width of a numbers
        for distance in distances:
            if distance.data > median_width:
                symbols.append(symbol_data(" ", distance.pos))

        # sort the list by the x position again to place the spaces at the correct position
        symbols = list(sorted(symbols, key=lambda sym: sym.pos.x))

        # create a string from the symbols
        value = "".join(map(lambda sym: sym.data, symbols))
        # create numbers for every number in the string, split by a space
        numbers = tuple([float(s) for s in value.split(" ")])
        return (numbers, img)


if __name__ == "__main__":
    img = cv.imread("sample_image.PNG")
    repetitions = 1
    t = process_time()
    num_reader = number_reader()
    for i in range(0, repetitions):
        print(num_reader.read_number_from_img(img)[0])
    t = process_time() - t
    print(t * 1000, t/repetitions * 1000)
