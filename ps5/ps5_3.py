import numpy as np
import os
import math
import cv2
import opticalflow
import pyramids

def get_op_result(frame1, frame2, n, filter_size):
    frame1_pyr_lst = pyramids.gaussian_pyramids_gen(frame1, 4)
    frame2_pyr_lst = pyramids.gaussian_pyramids_gen(frame2, 4)

    width = frame1_pyr_lst[n].shape[1]
    height = frame1_pyr_lst[n].shape[0]

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)

    op_flow = opticalflow.calculate_LK(frame1_pyr_lst[n], frame2_pyr_lst[n], filter_size, 15)

    xv = xv + op_flow[..., 0]
    yv = yv + op_flow[..., 1]

    warped_img = cv2.remap(frame1_pyr_lst[n], xv.astype(np.float32), yv.astype(np.float32), cv2.INTER_LINEAR)

    expanded_warped_img = warped_img
    expanded_shift_img = frame2_pyr_lst[n]

    for i in range(0, n):
        expanded_warped_img = pyramids.expand(expanded_warped_img)
        expanded_shift_img = pyramids.expand(expanded_shift_img)

    diff_img = expanded_warped_img - expanded_shift_img
    img_u = np.zeros(op_flow[..., 0].shape)
    img_v = np.zeros(op_flow[..., 1].shape)

    img_u = cv2.normalize(op_flow[..., 0], img_u, norm_type=cv2.NORM_MINMAX)
    img_v = cv2.normalize(op_flow[..., 1], img_v, norm_type=cv2.NORM_MINMAX)

    return img_u, img_v, diff_img

def run_data_seq(filename, dataSeq, n, filter_size):
    for i in range(1, len(dataSeq)):
        frame1 = dataSeq[i - 1]
        frame2 = dataSeq[i]
        img_u, img_v, diff_img = get_op_result(frame1, frame2, n, filter_size)
        disp_img = np.concatenate((img_u, img_v), axis=0)
        cv2.imwrite(os.path.join('output', filename + '-1-{0}.png'.format(i - 1)), disp_img * 255)
        cv2.imwrite(os.path.join('output', filename + '-2-{0}.png'.format(i - 1)), diff_img)
    return

dataSeq1 = []
dataSeq2 = []

dataSeq1.append(cv2.imread(os.path.join('input\\DataSeq1', 'yos_img_01.jpg'),0))
dataSeq1.append(cv2.imread(os.path.join('input\\DataSeq1', 'yos_img_02.jpg'),0))
dataSeq1.append(cv2.imread(os.path.join('input\\DataSeq1', 'yos_img_03.jpg'),0))

run_data_seq('ps5-3-a', dataSeq1, 1, (19, 19))

dataSeq2.append(cv2.imread(os.path.join('input\\DataSeq2', '0.png'),0))
dataSeq2.append(cv2.imread(os.path.join('input\\DataSeq2', '1.png'),0))
dataSeq2.append(cv2.imread(os.path.join('input\\DataSeq2', '2.png'),0))

run_data_seq('ps5-3-b', dataSeq2, 3, (5, 5))