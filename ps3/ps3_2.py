import os
import numpy as np
import cv2
import load_points as lp
import Calibration as calib
import calc
import test
import utility

ptsListPathA = os.path.join('input', 'pts2d-pic_a.txt')
ptsListPathB = os.path.join('input', 'pts2d-pic_b.txt')

ptsArrA = lp.LoadPoints2D(ptsListPathA)
ptsArrB = lp.LoadPoints2D(ptsListPathB)

normPtsArrA, Ta = calc.Normalize2DPts(ptsArrA)
normPtsArrB, Tb = calc.Normalize2DPts(ptsArrB)

#print normPtsArrA
#print normPtsArrB

rawF = calib.LSQrawFSolver(ptsArrA, ptsArrB)
normRawF = calib.LSQrawFSolver(normPtsArrA, normPtsArrB)
#print rawF

F= calib.LinearFfixer(rawF)
normF = calib.LinearFfixer(normRawF)

F2 = np.dot(np.dot(Tb.T,normF),Ta)
F2 = F2/F2[2,2]

print F
print F2

img_a = cv2.imread(os.path.join('input', 'pic_a.jpg'))
img_b = cv2.imread(os.path.join('input', 'pic_b.jpg'))

#print img_a.shape[0]

lineArr = calib.GetEpipolarLines(img_a, ptsArrB, F2.T)

for i in range(len(lineArr[:, 0])):
    cv2.line(img_a, tuple(lineArr[i, 0].astype(int)), tuple(lineArr[i, 1].astype(int)), (255,0,0))

lineArr = calib.GetEpipolarLines(img_b, ptsArrA, F2)

for i in range(len(lineArr[:, 0])):
    cv2.line(img_b, tuple(lineArr[i, 0].astype(int)), tuple(lineArr[i, 1].astype(int)), (255,0,0))


cv2.imshow('PicA', img_a)
cv2.imshow('PicB', img_b)

cv2.imwrite(os.path.join('output', 'ps3-2-c-3.png'),img_a)
cv2.imwrite(os.path.join('output', 'ps3-2-c-4.png'),img_b)
cv2.waitKey(0)
