import numpy as np
import os
import math
import cv2
import opticalflow

def calc_and_show_lk_op_flow(frame1, frame2, save_name):
    op_flow = opticalflow.calculate_LK(frame1, frame2, (31,31), 50)
    opticalflow.vis_optic_flow_arrows(frame1, op_flow, os.path.join('output', save_name), show=False)
    return

original_img = cv2.imread(os.path.join('input\TestSeq', 'Shift0.png'),0)
shiftR2_img = cv2.imread(os.path.join('input\TestSeq', 'ShiftR2.png'),0)
shiftR5U5_img = cv2.imread(os.path.join('input\TestSeq', 'ShiftR5U5.png'),0)
shiftR10_img = cv2.imread(os.path.join('input\TestSeq', 'ShiftR10.png'),0)
shiftR20_img = cv2.imread(os.path.join('input\TestSeq', 'ShiftR20.png'),0)
shiftR40_img = cv2.imread(os.path.join('input\TestSeq', 'ShiftR40.png'),0)

calc_and_show_lk_op_flow(original_img, shiftR2_img, 'ps5-1-a-1.png')
calc_and_show_lk_op_flow(original_img, shiftR5U5_img, 'ps5-1-a-2.png')
calc_and_show_lk_op_flow(original_img, shiftR10_img, 'ps5-1-b-1.png')
calc_and_show_lk_op_flow(original_img, shiftR20_img, 'ps5-1-b-2.png')
calc_and_show_lk_op_flow(original_img, shiftR40_img, 'ps5-1-b-3.png')

