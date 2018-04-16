import numpy as np
import cv2

def stochastic_universal_sampling(weight_vec):
    sample_set = []
    cdf = []
    n = len(weight_vec)
    cdf.append(weight_vec[0][2])
    for i in range(1, n):
        cdf.append(cdf[i - 1] + weight_vec[i][2])

    u = 1.0/n
    i = 0
    for j in range(0, n):
        while (u > cdf[i]):
            if i >= len(cdf) - 1:
                break
            i = i + 1
        sample_set.append([weight_vec[i][0], weight_vec[i][1], 1.0/n])
        u = u + 1.0/n

    return sample_set

def mean_squared_error(template, target):
    mse = np.sum(((template.astype(float) - target.astype(float)) ** 2)) / (template.shape[0] * template.shape[1])
    return mse

def histogram_distance(template, target):
    template_hist = np.zeros((256, 3))
    target_hist = np.zeros((256, 3))

    width = template.shape[1]
    height = template.shape[0]

    for ch in range(0, 3):
        template_hist[:, ch] = np.array(cv2.calcHist([template], [ch], None, [256], [0, 256]))[:, 0]
        target_hist[:, ch] = np.array(cv2.calcHist([target.astype(np.uint8)], [ch], None, [256], [0, 256]))[:, 0]

    template_hist /= float(width * height)
    target_hist /= float(width * height)

    dist = 0


    '''
    for k in range(0, 256):
        for ch in range(0, 3):
            if template_hist[k, ch] == 0 and target_hist[k, ch] == 0:
                continue
            else:
                dist += (template_hist[k, ch] - target_hist[k, ch])**2/(template_hist[k, ch] + target_hist[k, ch])
    '''
    hist_sum = template_hist + target_hist
    nonzero_mask = hist_sum > 0
    hist_sub = template_hist - target_hist

    dist = 0.5 * np.sum(hist_sub[nonzero_mask]**2 / hist_sum[nonzero_mask])

    return dist