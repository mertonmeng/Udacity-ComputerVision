import os
import numpy as np
import cv2
import load_points as lp
import Calibration as calib
import calc
import test

ptsListPathA = os.path.join('input', 'pts2d-pic_a.txt')
ptsListPathB = os.path.join('input', 'pts2d-pic_b.txt')

ptsArrA = lp.LoadPoints2D(ptsListPathA)
ptsArrB = lp.LoadPoints2D(ptsListPathB)

rawF = calib.LSQrawFSolver(ptsArrA, ptsArrB)
#print rawF

F= calib.LinearFfixer(rawF)
print F

F2 = test.svd_F_solver(ptsArrA, ptsArrB)
#print F2

F2= calib.LinearFfixer(F2)
print F2