import numpy as np
import cv2

def disparity_ncorr(L, R):
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
            curStrip = Padded_R[i - halfWinSize : i + halfWinSize, j - halfWinSize - distMax: j + halfWinSize + distMax]
            res = cv2.matchTemplate(curStrip.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            max_col = max_loc[0] + j - distMax
            dist = max_col - j
            D[i, j] = dist

    # TODO: Your code here

    return D[distMax + halfWinSize: len(D[:,1]) - distMax - halfWinSize, distMax + halfWinSize: len(D[1,:]) - distMax - halfWinSize]

    # TODO: Your code here
