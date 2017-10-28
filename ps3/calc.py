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