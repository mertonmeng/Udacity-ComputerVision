import numpy as np

def SVDMSolver(pts3D, pts2D):

    vecList = []
    for i in range(0,len(pts3D[:,0])):
        vec1 = np.array([pts3D[i, 0], pts3D[i, 1], pts3D[i, 2], 1.0, 0.0, 0.0, 0.0, 0.0,
                -pts2D[i, 0] * pts3D[i, 0], -pts2D[i, 0] * pts3D[i, 1], -pts2D[i, 0] * pts3D[i, 2], -pts2D[i, 0]])
        vec2 = np.array([0.0, 0.0, 0.0, 0.0, pts3D[i, 0], pts3D[i, 1], pts3D[i, 2], 1.0,
                -pts2D[i, 1] * pts3D[i, 0], -pts2D[i, 1] * pts3D[i, 1], -pts2D[i, 1] * pts3D[i, 2], -pts2D[i, 1]])
        vecList.append(vec1)
        vecList.append(vec2)

    A = np.array(vecList, dtype=float)
    U, s, VTrans = np.linalg.svd(A)

    V = VTrans.T
    MVec = V[:, len(V[0, :]) - 1]
    M = np.array([MVec[0:4], MVec[4:8], MVec[8:12]])

    return M

def LSQrawFSolver(ptsArrA, ptsArrB):

    AList = []

    for i in range(0, len(ptsArrA[:, 0])):
        u1 = ptsArrA[i, 0]
        v1 = ptsArrA[i, 1]
        u2 = ptsArrB[i, 0]
        v2 = ptsArrB[i, 1]
        vec1 = np.array([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1], dtype = float)
        AList.append(vec1)

    A = np.array(AList)
    #print A
    Fvec,_,_,_ = np.linalg.lstsq(A, -np.ones(A.shape[0]).T)
    Fvec = np.append(Fvec,1)
    F = np.reshape(np.array(Fvec), (3,3))

    return F

def LinearFfixer(rawF):
    U, s, VTrans = np.linalg.svd(rawF)
    D = np.diag(s)
    D[D.shape[0] - 1, D.shape[1] - 1] = 0
    F = np.dot(U, np.dot(D, VTrans))
    return F

def GetEpipolarLines(img, points2DArr, F):

    lineList = []

    ptTL = np.array([0,0,1], dtype=float)
    ptTR = np.array([img.shape[1] - 1, 0, 1], dtype=float)
    ptBL = np.array([0, img.shape[0] - 1, 1], dtype=float)
    ptBR = np.array([img.shape[1] - 1, img.shape[0] - 1, 1], dtype=float)

    lineL = np.cross(ptTL, ptBL)
    lineR = np.cross(ptTR, ptBR)

    print lineL
    print lineR

    for i in range(0, len(points2DArr[:, 0])):
        pointVec = np.array([points2DArr[i , 0], points2DArr[i , 1], 1], dtype=float)
        epiLineVec = np.dot(F, pointVec)
        epiLinePtL = np.cross(epiLineVec, lineL)
        epiLinePtR = np.cross(epiLineVec, lineR)
        lineList.append([[epiLinePtL[0] / epiLinePtL[2], epiLinePtL[1] / epiLinePtL[2]],
                         [epiLinePtR[0] / epiLinePtR[2], epiLinePtR[1] / epiLinePtR[2]]])

    lineArr = np.array(lineList)

    return lineArr