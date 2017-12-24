import numpy as np
import cv2
from scipy import signal

def generate_wkernel(a):
    w_vec = np.array([0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2], dtype=np.float32)
    w_kernel = np.outer(w_vec.T, w_vec)
    return w_kernel

def reduce(img):
    w = generate_wkernel(0.4)
    filtered_img = signal.convolve2d(img, w, mode='same', boundary='symm')
    reduced_img = filtered_img[0:-1:2,0:-1:2]
    return reduced_img

def expand(img):
    width = img.shape[1]
    height = img.shape[0]
    expanded_img = np.zeros((2 * height - 1, 2 * width - 1))
    w_kernel = generate_wkernel(0.4)
    for i in range(2 * height - 1):
        for j in range(2 * width - 1):
            wsum = 0
            for m in range(-2, 3):
                for n in range(-2, 3):
                    if (i - m) % 2 != 0 or (j - n) % 2 != 0:
                        continue
                    r = 0
                    c = 0
                    if i - m >= 0 and i - m < 2 * height - 1:
                        r = (i - m) / 2
                    elif i - m < 0:
                        r = 0
                    else:
                        r = height - 1

                    if j - n >= 0 and j - n < 2 * width - 1:
                        c = (j - n) / 2
                    elif j - n < 0:
                        c = 0
                    else:
                        c = width - 1

                    wsum += w_kernel[m + 2, n + 2] * img[r, c]
            expanded_img[i, j] = 4 * wsum

    return expanded_img

def gaussian_pyramids_gen(img, n):

    pyramids_lst = []
    last_layer = img
    pyramids_lst.append(last_layer)

    for i in range(n - 1):
        last_layer = reduce(last_layer)
        pyramids_lst.append(last_layer)

    return pyramids_lst

def laplacian_pyramids_gen(gau_pyramids_lst):
    lap_pyramids_lst = []
    original_img = gau_pyramids_lst[0]
    lap_pyramids_lst.append(original_img)
    last_expanded_img = original_img

    for i in range(1, len(gau_pyramids_lst)):
        cur_gau_img = gau_pyramids_lst[i]
        for j in range(0, i):
            cur_gau_img = expand(cur_gau_img)

        resized_img = last_expanded_img[0:cur_gau_img.shape[0], 0:cur_gau_img.shape[1]]
        cur_lap_img = cv2.subtract(resized_img, cur_gau_img.astype(np.uint8))
        lap_pyramids_lst.append(cur_lap_img)
        last_expanded_img = cur_gau_img.astype(np.uint8)

    return lap_pyramids_lst