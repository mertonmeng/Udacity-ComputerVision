import numpy as np
import random

def LoadPoints2D(path):
    pointList = []
    file = open(path, "r")
    for line in file:
        coordStr = line.split()
        pt = [float(coordStr[0]), float(coordStr[1])]
        pointList.append(pt)

    pointArr = np.array(pointList)
    return pointArr

def LoadPoints3D(path):
    pointList = []
    file = open(path, "r")
    for line in file:
        coordStr = line.split()
        pt = [float(coordStr[0]), float(coordStr[1]), float(coordStr[2])]
        pointList.append(pt)

    pointArr = np.array(pointList)
    return pointArr

def GetRandomPointArray(ptsList2D, pointListLen, ptsList3D):

    randpts2DList = []
    randpts3DList = []

    while len(randpts2DList) < pointListLen:
        ptIdx = random.randint(0, len(ptsList2D) - 1)
        pt2D = ptsList2D.pop(ptIdx)
        pt3D = ptsList3D.pop(ptIdx)
        randpts2DList.append(pt2D)
        randpts3DList.append(pt3D)

    randpts2DArr = np.array(randpts2DList)
    randpts3DArr = np.array(randpts3DList)

    return randpts2DArr,randpts3DArr