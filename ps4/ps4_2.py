import numpy as np
import cv2
import utility as util
import os
from scipy import signal
from scipy import ndimage

def run_main(filename):
    gaussian_filter_kernel = (5, 5)
    img = cv2.imread(os.path.join('input', filename), 0)
    gx, gy, gx_norm, gy_norm = util.get_gradient_img(img, gaussian_filter_kernel, 2)
    angle_map, angle_map_norm = util.get_angle_map(gx, gy)

    harris_filter_kernel = (5, 5)
    r, r_norm = util.get_harris_value(gx, gy, harris_filter_kernel)
    corner_pts = util.find_corner_point(r, 5, 0.01)
    img_corner = util.draw_points(img, corner_pts)

    key_pts_list = util.get_key_points(corner_pts, angle_map)
    image_keypoints = util.draw_interest_points(img, key_pts_list)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_sift, descriptors = sift.compute(img, key_pts_list)

    return keypoints_sift, descriptors, image_keypoints

img_sim_a = cv2.imread(os.path.join('input', 'simA.jpg'),0)
img_trans_a = cv2.imread(os.path.join('input', 'transA.jpg'),0)
img_sim_b = cv2.imread(os.path.join('input', 'simB.jpg'),0)
img_trans_b = cv2.imread(os.path.join('input', 'transB.jpg'),0)

sim_a_kpts, sim_a_descriptors, img_sim_a_color = run_main('simA.jpg')
trans_a_kpts, trans_a_descriptors, img_sim_b_color = run_main('transA.jpg')
sim_b_kpts, sim_b_descriptors, img_trans_a_color = run_main('simB.jpg')
trans_b_kpts, trans_b_descriptors, img_trans_b_color = run_main('transB.jpg')

stacked_sim_img = np.concatenate((img_sim_a_color, img_sim_b_color), axis=1)
stacked_trans_img = np.concatenate((img_trans_a_color, img_trans_b_color), axis=1)

#cv2.imwrite(os.path.join('output', 'ps4-2-a-1.png'), stacked_sim_img)
#cv2.imwrite(os.path.join('output', 'ps4-2-a-2.png'), stacked_trans_img)

matches_sim, match_img_sim = util.get_matches(img_sim_a, img_sim_b, sim_a_kpts, sim_b_kpts, sim_a_descriptors, sim_b_descriptors)
matches_trans, match_img_trans = util.get_matches(img_trans_a, img_trans_b, trans_a_kpts, trans_b_kpts, trans_a_descriptors, trans_b_descriptors)

cv2.imwrite(os.path.join('output', 'ps4-2-b-1.png'), match_img_trans)
cv2.imwrite(os.path.join('output', 'ps4-2-b-2.png'), match_img_sim)
