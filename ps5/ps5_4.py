import numpy as np
import os
import math
import cv2
import opticalflow
import pyramids

def get_op_result(frame1, frame2, n, half_filter):
    frame1_pyr_lst = pyramids.gaussian_pyramids_gen(frame1, n)
    frame2_pyr_lst = pyramids.gaussian_pyramids_gen(frame2, n)

    img_u = np.zeros(frame1_pyr_lst[-1].shape)
    img_v = np.zeros(frame1_pyr_lst[-1].shape)

    warped_img = np.zeros(frame1_pyr_lst[0].shape)

    for i in range(0, n):

        cur_layer_img1 = frame1_pyr_lst[n - 1 - i]
        cur_layer_img2 = frame2_pyr_lst[n - 1 - i]

        if i != 0:
            img_u = pyramids.expand(img_u) * 2
            img_v = pyramids.expand(img_v) * 2
            if img_u.shape != cur_layer_img1.shape:
                padding_u = np.zeros(cur_layer_img1.shape)
                padding_u[0:img_u.shape[0], 0:img_u.shape[1]] = img_u
                img_u = padding_u
                padding_v = np.zeros(cur_layer_img1.shape)
                padding_v[0:img_v.shape[0], 0:img_v.shape[1]] = img_v
                img_v = padding_v

        width = cur_layer_img1.shape[1]
        height = cur_layer_img1.shape[0]

        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        xv, yv = np.meshgrid(x, y)

        xv = xv + img_u
        yv = yv + img_v

        warped_img = cv2.remap(cur_layer_img1, xv.astype(np.float32), yv.astype(np.float32), cv2.INTER_LINEAR)

        filter_shape = (2 * half_filter - 1, 2 * half_filter - 1)

        op_flow = opticalflow.calculate_LK(warped_img, cur_layer_img2, filter_shape, 10)

        half_filter = half_filter * 2
        img_u = img_u + op_flow[..., 0]
        img_v = img_v + op_flow[..., 1]

    return warped_img, img_u, img_v

def run_data_seq(filename, dataSeq, n, filter_size):
    for i in range(1, len(dataSeq)):
        frame1 = dataSeq[i - 1]
        frame2 = dataSeq[i]
        warped_img, img_u, img_v = get_op_result(frame1, frame2, n, filter_size)
        diff_img = frame2.astype(np.float32) - warped_img
        img_u = cv2.normalize(img_u, img_u, norm_type=cv2.NORM_MINMAX)
        img_v = cv2.normalize(img_v, img_v, norm_type=cv2.NORM_MINMAX)
        disp_img = np.concatenate((img_u, img_v), axis=0)
        cv2.imwrite(os.path.join('output', filename + '-1-{0}.png'.format(i - 1)), disp_img * 255)
        cv2.imwrite(os.path.join('output', filename + '-2-{0}.png'.format(i - 1)), diff_img)
    return

dataSeq1 = []
dataSeq2 = []
testSeq = []

testSeq.append(cv2.imread(os.path.join('input\TestSeq', 'Shift0.png'), 0))
testSeq.append(cv2.imread(os.path.join('input\TestSeq', 'ShiftR2.png'), 0))
testSeq.append(cv2.imread(os.path.join('input\TestSeq', 'ShiftR5U5.png'), 0))
testSeq.append(cv2.imread(os.path.join('input\TestSeq', 'ShiftR10.png'), 0))
testSeq.append(cv2.imread(os.path.join('input\TestSeq', 'ShiftR20.png'), 0))
testSeq.append(cv2.imread(os.path.join('input\TestSeq', 'ShiftR40.png'), 0))

run_data_seq('ps5-4-a', testSeq, 4, 3)

dataSeq1.append(cv2.imread(os.path.join('input\\DataSeq1', 'yos_img_01.jpg'), 0))
dataSeq1.append(cv2.imread(os.path.join('input\\DataSeq1', 'yos_img_02.jpg'), 0))
dataSeq1.append(cv2.imread(os.path.join('input\\DataSeq1', 'yos_img_03.jpg'), 0))

run_data_seq('ps5-4-b', dataSeq1, 4, 3)

dataSeq2.append(cv2.imread(os.path.join('input\\DataSeq2', '0.png'), 0))
dataSeq2.append(cv2.imread(os.path.join('input\\DataSeq2', '1.png'), 0))
dataSeq2.append(cv2.imread(os.path.join('input\\DataSeq2', '2.png'), 0))

run_data_seq('ps5-4-c', dataSeq2, 4, 3)