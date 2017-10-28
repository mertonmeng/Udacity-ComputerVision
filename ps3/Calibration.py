import numpy as np

def NormCalibrate(pts3D, pts2D):

    vecList = []
    for i in range(0,len(pts3D[:,0])):
        vec1 = np.array([pts3D[i, 0], pts3D[i, 1], pts3D[i, 2], 1.0, 0.0, 0.0, 0.0, 0.0,
                -pts2D[i, 0] * pts3D[i, 0], -pts2D[i, 0] * pts3D[i, 1], -pts2D[i, 0] * pts3D[i, 2], -pts2D[i, 0]])
        vec2 = np.array([0.0, 0.0, 0.0, 0.0, pts3D[i, 0], pts3D[i, 1], pts3D[i, 2], 1.0,
                -pts2D[i, 1] * pts3D[i, 0], -pts2D[i, 1] * pts3D[i, 1], -pts2D[i, 1] * pts3D[i, 2], -pts2D[i, 1]])
        vecList.append(vec1)
        vecList.append(vec2)

    A = np.array(vecList, dtype=float)
    U, D, VTrans = np.linalg.svd(A)

    V = VTrans.T
    MVec = V[:, len(V[0, :]) - 1]
    M = np.array([MVec[0:4], MVec[4:8], MVec[8:12]])

    return M