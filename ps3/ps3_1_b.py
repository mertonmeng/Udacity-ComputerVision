import os
import numpy as np
import cv2
import load_points as lp
import Calibration as calib
import calc
import copy
#import test

norm2DPtPathA = os.path.join('input', 'pts2d-pic_b.txt')
norm3DPtPath = os.path.join('input', 'pts3d.txt')

norm2DPtsA = lp.LoadPoints2D(norm2DPtPathA)
norm3DPts = lp.LoadPoints3D(norm3DPtPath)

norm2DPtsAList = norm2DPtsA.tolist()
norm3DPtsList = norm3DPts.tolist()

minResidual = float("Inf")
bestM = np.zeros((3,4))

for i in range(0,10):
    copyList2D = copy.copy(norm2DPtsAList)
    copyList3D = copy.copy(norm3DPtsList)
    randPts2D,randPts3D = lp.GetRandomPointArray(copyList2D, 16, copyList3D)
    M = calib.NormCalibrate(randPts3D, randPts2D)
    restPts2D = np.array(copyList2D)
    restPts3D = np.array(copyList3D)
    residual = calc.GetResidual(restPts2D, M, restPts3D)
    if residual.sum() < minResidual:
        bestM = M
        minResidual = residual.sum()


Q = bestM[0:3, 0:3]
m4 = bestM[:, 3]
C = np.dot(-np.linalg.inv(Q), m4)

print C
#print minResidual