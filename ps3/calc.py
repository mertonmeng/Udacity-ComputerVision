import numpy as np
import math

def GetResidual(pts2D, M, pts3D):

    residualList = []

    for i in range(0, len(pts3D[:, 0])):
        pt3D = np.array([pts3D[i, 0], pts3D[i, 1], pts3D[i, 2], 1])
        pt2Dcomputed = np.dot(M,pt3D.T)
        pt2Dcomputed /= pt2Dcomputed[2]
        pt2D = np.append(pts2D[i],[1]).T
        residualList.append(math.sqrt(((pt2D - pt2Dcomputed)**2).sum()))

    residual = np.array(residualList)

    return residual

def Normalize2DPts(pts2D):

    normPtsList = []

    mean_u = np.mean(pts2D[:,0])
    mean_v = np.mean(pts2D[:,1])
    scale_factor = 1/np.max(np.abs(pts2D))

    scale_mat = np.diag([scale_factor, scale_factor, 1])
    offset_mat = np.diag([1, 1, 1])
    offset_mat[0, 2] = -mean_u
    offset_mat[1, 2] = -mean_v
    T = np.dot(scale_mat, offset_mat)

    for i in range(0, len(pts2D[:, 0])):
        pointVec = np.array([pts2D[i , 0], pts2D[i , 1], 1], dtype=float)
        normPts = np.dot(T, pointVec.T)
        normPtsList.append([normPts[0], normPts[1]])

    normPtsArr = np.array(normPtsList)

    return normPtsArr,T