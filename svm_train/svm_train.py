from cv2 import ml
import cv2 as cv
import numpy as np
import glob
import zipfile
import shutil

trainCells = []
testCells = []
trainLabels = []
testLabels = []
trainHOG = []
testHOG = []


def loadTrainTestLabels():
    imageNumbers = []

    for i in range(10):
        images = glob.glob(f"./train_test_images/{i}*.png")
        imageNumbers.append((i, len(images)))
        for j in range(len(images)):
            img = cv.imread(images[j], cv.IMREAD_GRAYSCALE)
            if(j < 0.9 * len(images)):
                trainCells.append(img)
                trainLabels.append(i)
            else:
                testCells.append(img)
                testLabels.append(i)


winSize = (20, 20)  # (6, 12)
cellSize = (8, 8)  # (6, 12)
blockSize = (8, 8)  # (6, 12)
blockStride = (4, 4)  # (6, 6)
nbins = 9
signedGradients = True
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64

hog = cv.HOGDescriptor(
    winSize,
    blockSize,
    blockStride,
    cellSize,
    nbins,
    derivAperture,
    winSigma,
    histogramNormType,
    L2HysThreshold,
    gammaCorrection,
    nlevels,
    signedGradients
)


def createTestTrainHOG():
    for i in trainCells:
        descriptor = hog.compute(i)
        trainHOG.append(descriptor)
    for i in testCells:
        descriptor = hog.compute(i)
        testHOG.append(descriptor)


def trainSVM():
    svm = ml.SVM_create()
    svm.setType(ml.SVM_C_SVC)
    svm.setKernel(ml.SVM_RBF)
    td = ml.TrainData_create(trainData, ml.ROW_SAMPLE, np.array(trainLabels))
    # svm.train(td)
    svm.trainAuto(trainData, ml.ROW_SAMPLE, np.array(trainLabels))
    svm.save("../svm_data.yaml")
    return svm


# extract all images from the zip folder
with zipfile.ZipFile('train_test_images.zip') as small:
    small.extractall('train_test_images/')

loadTrainTestLabels()
createTestTrainHOG()

trainData = np.float32(trainHOG)
testData = np.float32(testHOG)
svm = trainSVM()

result = svm.predict(testData, np.array(testLabels))


good = 0
bad = 0

for i in range(len(testLabels)):
    if testLabels[i] == result[1][i]:
        good += 1
        print(str(testLabels[i]) + " pass")
    else:
        bad += 1
        print(str(testLabels[i]) + " fail " + str(result[1][i]))


print(good * 100.0 / (good + bad))

# delete the extracted images
shutil.rmtree('train_test_images/')
