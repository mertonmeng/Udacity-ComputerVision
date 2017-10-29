import os
import numpy as np
import cv2
import load_points as lp
import Calibration as calib
import calc
#import test

norm2DPtPathA = os.path.join('input', 'pts2d-norm-pic_a.txt')
norm3DPtPath = os.path.join('input', 'pts3d-norm.txt')

norm2DPtsA = lp.LoadPoints2D(norm2DPtPathA)
norm3DPts = lp.LoadPoints3D(norm3DPtPath)

norm2DPtsAList = norm2DPtsA.tolist()
norm3DPtsList = norm3DPts.tolist()

#rand2DPts,rand3DPts = lp.GetRandomPointArray(norm2DPtsAList, 8, norm3DPtsList)

#M2 = test.svd_M_solver(norm2DPtsA, norm3DPts)
M = calib.SVDMSolver(norm3DPts, norm2DPtsA)
residual = calc.GetResidual(norm2DPtsA, M, norm3DPts)

Q = M[0:3,0:3]
m4 = M[:,3]

C = np.dot(-np.linalg.inv(Q), m4)

print C
#print residual