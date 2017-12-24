import numpy as np
import sys
import math
import cv2
import copy
from scipy import signal
import matplotlib.pyplot as plt

def calculate_LK(frame1, frame2, filter_size = (31,31), thresh = 50):
    img_u = np.zeros(frame1.shape)
    img_v = np.zeros(frame2.shape)

    #frame1 = cv2.GaussianBlur(frame1, filter_size, 2)
    #frame2 = cv2.GaussianBlur(frame2, filter_size, 2)

    gy, gx = np.gradient(frame1)

    gt = frame1 - frame2

    gxx = gx * gx
    gxy = gx * gy
    gyy = gy * gy

    gxt = gx * gt
    gyt = gy * gt

    gxx_wsum = cv2.boxFilter(gxx, -1, ksize = filter_size, normalize = True)
    gxy_wsum = cv2.boxFilter(gxy, -1, ksize = filter_size, normalize = True)
    gyy_wsum = cv2.boxFilter(gyy, -1, ksize = filter_size, normalize = True)

    gxt_wsum = cv2.boxFilter(gxt, -1, ksize = filter_size, normalize = True)
    gyt_wsum = cv2.boxFilter(gyt, -1, ksize = filter_size, normalize = True)

    for i in range(0, gxx_wsum.shape[0]):
        for j in range(0, gxx_wsum.shape[1]):
            A = np.array([[gxx_wsum[i, j], gxy_wsum[i, j]], [gxy_wsum[i, j], gyy_wsum[i, j]]])
            det = np.linalg.det(A)
            if det != 0:
                b = np.array([- gxt_wsum[i, j], - gyt_wsum[i, j]]).T
                A_inv = np.linalg.inv(A)
                d = np.dot(A_inv, b)
                if abs(d[0]) < thresh:
                    img_u[i, j] = d[0]
                if abs(d[1]) < thresh:
                    img_v[i, j] = d[1]

    op_flow = np.zeros(img_u.shape + (2,))
    op_flow[..., 0] = img_u
    op_flow[..., 1] = img_v

    return op_flow

def lk_optic_flow(frame1, frame2, win=2):
    '''
    The code below was borrowed from stackoverflow
    ../questions/14321092/lucas-kanade-python-numpy-implementation-uses-enormous-amount-of-memory
    '''

    # calculate gradients in x, y and t dimensions
    Ix = np.zeros(frame1.shape, dtype=np.float32)
    Iy = np.zeros(frame1.shape, dtype=np.float32)
    It = np.zeros(frame1.shape, dtype=np.float32)
    Ix[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 2:], frame1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = cv2.subtract(frame1[2:, 1:-1], frame1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 1:-1], frame2[1:-1, 1:-1])

    params = np.zeros(frame1.shape + (5,))
    params[..., 0] = Ix ** 2
    params[..., 1] = Iy ** 2
    params[..., 2] = Ix * Iy
    params[..., 3] = Ix * It
    params[..., 4] = Iy * It
    del It, Ix, Iy
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(frame1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2

    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)
    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    op_flow = op_flow.astype(np.float32)
    return op_flow

def vis_optic_flow_arrows(img, flow, filename, show=True):
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)
    plt.figure()
    fig = plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    step = img.shape[0] / 50
    plt.quiver(x[::step, ::step], y[::step, ::step],
               flow[::step, ::step, 0], flow[::step, ::step, 1],
               color='r', pivot='middle', headwidth=2, headlength=3)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()