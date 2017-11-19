import numpy as np
import cv2
import utility as util
import os
from scipy import signal
from scipy import ndimage


img_sim_a = cv2.imread(os.path.join('input', 'simA.jpg'),0)
img_trans_a = cv2.imread(os.path.join('input', 'transA.jpg'),0)
img_sim_b = cv2.imread(os.path.join('input', 'simB.jpg'),0)
img_trans_b = cv2.imread(os.path.join('input', 'transB.jpg'),0)
img_check_a = cv2.imread(os.path.join('input', 'check_rot.bmp'),0)

gaussian_filter_kernel = (5,5)

gx_sim_a, gy_sim_a, gx_sim_a_norm, gy_sim_a_norm = util.get_gradient_img(img_sim_a, gaussian_filter_kernel, 2)
gx_trans_a, gy_trans_a, gx_trans_a_norm, gy_trans_a_norm = util.get_gradient_img(img_trans_a, gaussian_filter_kernel, 2)
gx_sim_b, gy_sim_b, gx_sim_b_norm, gy_sim_b_norm = util.get_gradient_img(img_sim_b, gaussian_filter_kernel, 2)
gx_trans_b, gy_trans_b, gx_trans_b_norm, gy_trans_b_norm = util.get_gradient_img(img_trans_b, gaussian_filter_kernel, 2)
gx_check_a, gy_check_a, gx_check_a_norm, gy_check_a_norm = util.get_gradient_img(img_check_a, gaussian_filter_kernel, 2)

stacked_g_img_sim_a = np.concatenate((gx_sim_a_norm, gy_sim_a_norm), axis=1)
stacked_g_img_trans_a = np.concatenate((gx_trans_a_norm, gy_trans_a_norm), axis=1)
stacked_g_img_check_a = np.concatenate((gx_check_a_norm, gy_check_a_norm), axis=1)

cv2.imwrite(os.path.join('output', 'ps4-1-a-1.png'), stacked_g_img_sim_a*255)
cv2.imwrite(os.path.join('output', 'ps4-1-a-2.png'), stacked_g_img_trans_a*255)

r_check,r_check_norm = util.get_harris_value(gx_check_a, gy_check_a, (11,11))

gaussian_filter_kernel_harris = (5,5)

r_sim_a,r_sim_a_norm = util.get_harris_value(gx_sim_a, gy_sim_a, gaussian_filter_kernel_harris)
r_sim_b,r_sim_b_norm = util.get_harris_value(gx_sim_b, gy_sim_b, gaussian_filter_kernel_harris)
r_trans_a,r_trans_a_norm = util.get_harris_value(gx_trans_a, gy_trans_a, gaussian_filter_kernel_harris)
r_trans_b,r_trans_b_norm = util.get_harris_value(gx_trans_b, gy_trans_b, gaussian_filter_kernel_harris)

cv2.imwrite(os.path.join('output', 'ps4-1-b-3.png'), r_sim_a_norm*255)
cv2.imwrite(os.path.join('output', 'ps4-1-b-4.png'), r_sim_b_norm*255)
cv2.imwrite(os.path.join('output', 'ps4-1-b-1.png'), r_trans_a_norm*255)
cv2.imwrite(os.path.join('output', 'ps4-1-b-2.png'), r_trans_b_norm*255)

pt_sim_a_arr = util.find_corner_point(r_sim_a, 10, 0.01)
pt_sim_b_arr = util.find_corner_point(r_sim_b, 10, 0.01)
pt_trans_a_arr = util.find_corner_point(r_trans_a, 10, 0.01)
pt_trans_b_arr = util.find_corner_point(r_trans_b, 10, 0.01)

#print pt_check_arr
img_sim_a_color = util.draw_points(img_sim_a, pt_sim_a_arr)
img_sim_b_color = util.draw_points(img_sim_b, pt_sim_b_arr)
img_trans_a_color = util.draw_points(img_trans_a, pt_trans_a_arr)
img_trans_b_color = util.draw_points(img_trans_b, pt_trans_b_arr)

cv2.imwrite(os.path.join('output', 'ps4-1-c-3.png'), img_sim_a_color)
cv2.imwrite(os.path.join('output', 'ps4-1-c-4.png'), img_sim_b_color)
cv2.imwrite(os.path.join('output', 'ps4-1-c-1.png'), img_trans_a_color)
cv2.imwrite(os.path.join('output', 'ps4-1-c-2.png'), img_trans_b_color)

#cv2.imshow('image', stacked_g_img_sim_a)
#cv2.imshow('image2', r_sim_a_norm)
#cv2.imshow('image3', check_img_with_corner)
cv2.waitKey(0)