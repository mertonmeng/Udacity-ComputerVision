# ps2
import os
import numpy as np
import cv2

## 1-a
# Read images
L = cv2.imread(os.path.join('input', 'pair2-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv2.imread(os.path.join('input', 'pair2-R.png'), 0) * (1.0 / 255.0)

#cv2.imshow("image",L)
#cv2.waitKey(0)
# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)

noise = 0.05 * np.random.randn(len(L[:,1]), len(L[1,:]))

#L += noise
#L *= 1.1

from disparity_ncorr import disparity_ncorr
D_L = disparity_ncorr(L, R)
D_R = disparity_ncorr(R, L)

#np.savetxt('test.csv', D_L, delimiter=',',fmt='%1.4f')

D_L = np.abs(D_L)/(D_L.max() - D_L.min())*255
D_R = np.abs(D_R)/(D_R.max() - D_R.min())*255



# TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly
cv2.imwrite(os.path.join('output', 'ps2-5-a-3.png'),D_L)
cv2.imwrite(os.path.join('output', 'ps2-5-a-4.png'),D_R)
# TODO: Rest of your code here
