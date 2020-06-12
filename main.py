import cv2 as cv
import numpy as np

class imageProcessor(object):

  def __init__(self, imageSource, colorEnv, blockResolution):
    tempImageSource = cv.imread(imageSource)
    self.imageData = tempImageSource
    self.imageResolution = tempImageSource.shape
    self.colorEnv = colorEnv
    self.blockResolution = blockResolution

  def getSpecificBlockData(self,line,column,returnPos):
    # topLeftPoint
    temp_xTLPos = column*self.blockResolution
    temp_yTLPos = line*self.blockResolution
    topLeftPoint = [int(temp_xTLPos),int(temp_yTLPos)]
    # bottomRightPoint
    temp_xBRPos = (column+1)*self.blockResolution
    temp_yBRPos = (line+1)*self.blockResolution
    bottomRightPoint = [int(temp_xBRPos),int(temp_yBRPos)]

    # pick block data
    if(returnPos):
        blockData = [(self.imageData[topLeftPoint[1]:bottomRightPoint[1], topLeftPoint[0]:bottomRightPoint[0]]),[topLeftPoint,bottomRightPoint]]
    else:
        blockData = self.imageData[topLeftPoint[1]:bottomRightPoint[1], topLeftPoint[0]:bottomRightPoint[0]]
    return blockData

class imageComparator(object):
    
  def __init__(self, imageData1, imageData2, threshold, blockResolution, maxColorDiff):

    self.imageData1 = cv.resize(imageData1,(blockResolution,blockResolution)) # Pattern
    self.imageData2 = imageData2

    self.maxColorDiff = maxColorDiff

    self.threshold = threshold
    self.blockResolution = blockResolution

  def color_distance(self, rgb1, rgb2):
      redDist = (int(rgb1[0]) - int(rgb2[0])) ** 2
      blueDist = (int(rgb1[1]) - int(rgb2[1])) ** 2
      greenDist = (int(rgb1[2]) - int(rgb2[2])) ** 2
      return (redDist+blueDist+greenDist)**0.5

  def getTwoBlocksSimilarities(self):
    pixNbr = self.blockResolution ** 2
    nbrSimPixs = 0
    for (imageData1_lineIndex, imageData1_line) in enumerate(self.imageData1):
        for (imageData1_pixelIndex, imageData1_pixel) in enumerate(imageData1_line):
            if(np.all(self.imageData2[imageData1_lineIndex][imageData1_pixelIndex] == imageData1_pixel) or self.color_distance(self.imageData2[imageData1_lineIndex][imageData1_pixelIndex],imageData1_pixel) < self.maxColorDiff):
                nbrSimPixs = nbrSimPixs+1
    preciseSim = nbrSimPixs/pixNbr
    isOverThreshold = False
    if(preciseSim >= self.threshold):
        isOverThreshold = True
    return ([isOverThreshold,preciseSim])
    


"""
fullLink = './images/full.png'
templateLink = './images/pattern.png'
param_blockResolution = 8
"""

fullLink = './images/mediumSize.jpg'
templateLink = './images/mediumPattern.jpg'
param_blockResolution = 8

fullImageSource = cv.imread(fullLink)
full = imageProcessor(fullLink,0,param_blockResolution)

pattern = imageProcessor(templateLink,0,param_blockResolution)
patternBlock = pattern.getSpecificBlockData(0,0,False)

thresholdMap = np.zeros(fullImageSource.shape, np.uint8)

nbrColumn = int(fullImageSource.shape[1]/param_blockResolution)
nbrLines = int(fullImageSource.shape[0]/param_blockResolution)



i=0
while i < nbrLines:
    z = 0
    while z < nbrColumn:
        testBlock = full.getSpecificBlockData(i,z,True)
        tempImageComparator = imageComparator(patternBlock, testBlock[0], 0.7, 8, 3)
        getTwoBlocksSimilaritiesWithPattern = tempImageComparator.getTwoBlocksSimilarities()
        imageThreshold = tempImageComparator.getTwoBlocksSimilarities()[1]
        imageThresholdBool = tempImageComparator.getTwoBlocksSimilarities()[0]
        tempRedVal = 255*(1-imageThreshold)
        tempBlueVal = 255*(imageThreshold)
        thresholdColor = (tempBlueVal,tempBlueVal,tempBlueVal)
        if(imageThresholdBool):
            thresholdColor = (0,255,0)
        cv.rectangle(thresholdMap,(testBlock[1][0][0],testBlock[1][0][1]),(testBlock[1][1][0],testBlock[1][1][1]),thresholdColor,-1)
        z = z+1
    i = i+1

fullWithThreshold = cv.addWeighted(fullImageSource, 0.25, thresholdMap, 0.75, 1)
#fullWithThresholdZoom = cv.resize(fullWithThreshold,(500,500), 0, 0, 0, 0)
filePath = "./gifRes/res_"
filePath += str(param_blockResolution)
filePath += "_.jpg"
cv.imwrite( filePath, fullWithThreshold);
cv.imshow("1",fullWithThreshold)
cv.waitKey()