import numpy as np

def disparity_ssd(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    distMax = 90
    halfWinSize = 6
    D = np.pad(np.zeros(L.shape),distMax + halfWinSize,'constant',constant_values=0)
    Padded_L = np.pad(L,distMax + halfWinSize,'constant',constant_values=0)
    Padded_R = np.pad(R, distMax + halfWinSize, 'constant', constant_values=0)

    for i in range(distMax + halfWinSize, len(D[:,1]) - distMax - halfWinSize):
        for j in range(distMax + halfWinSize, len(D[1,:]) - distMax - halfWinSize):
            template = Padded_L[i - halfWinSize: i + halfWinSize, j - halfWinSize: j + halfWinSize]
            min_diff = float("inf")
            best_x = 0
            for k in range(j - distMax, j + distMax):
                curWindow = Padded_R[i - halfWinSize: i + halfWinSize, k - halfWinSize: k + halfWinSize]
                diff = ((template - curWindow)**2).sum()
                if diff < min_diff:
                    min_diff = diff
                    best_x = k
            dist = best_x - j
            if np.abs(dist) > distMax:
                D[i, j] = distMax
            else:
                D[i, j] = dist

    # TODO: Your code here

    return D[distMax + halfWinSize: len(D[:,1]) - distMax - halfWinSize, distMax + halfWinSize: len(D[1,:]) - distMax - halfWinSize]