import numpy as np
import math
import os
from scipy import signal
import cv2
import utility as util

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

matches_sim, match_img_sim = util.get_matches(img_sim_a, img_sim_b, sim_a_kpts, sim_b_kpts, sim_a_descriptors, sim_b_descriptors)
matches_trans, match_img_trans = util.get_matches(img_trans_a, img_trans_b, trans_a_kpts, trans_b_kpts, trans_a_descriptors, trans_b_descriptors)

'''
best_trans, best_pair_list = util.get_ransac_translation(trans_a_kpts, trans_b_kpts, matches_trans, 100, 0.99)
best_match_img_trans = cv2.cvtColor(np.concatenate((img_trans_a, img_trans_b), axis=1),cv2.COLOR_GRAY2RGB)
trans_pt1 = (int(img_trans_a.shape[1]/2), int(img_trans_a.shape[0]/2))
trans_pt2 = (int(img_trans_a.shape[1]/2 + img_trans_a.shape[1] - best_trans[0]), int(img_trans_a.shape[0]/2 - best_trans[1]))
cv2.line(best_match_img_trans, trans_pt1, trans_pt2, (0, 255, 0), 5)
cv2.imwrite(os.path.join('output', 'ps4-3-a-1.png'), best_match_img_trans)
'''

best_match_img_sim = cv2.cvtColor(np.concatenate((img_sim_a, img_sim_b), axis=1),cv2.COLOR_GRAY2RGB)
best_match_img_affine = cv2.cvtColor(np.concatenate((img_sim_a, img_sim_b), axis=1),cv2.COLOR_GRAY2RGB)
best_trans_sim, best_pts_pair_sim = util.get_ransac_sim_trans(sim_a_kpts, sim_b_kpts, matches_sim, 50, 0.99)
best_trans_affine, best_pts_pair_affine = util.get_ransac_affine_trans(sim_a_kpts, sim_b_kpts, matches_sim, 50, 0.99)

'''
util.draw_pair(best_match_img_sim, best_pts_pair_sim)
util.draw_pair(best_match_img_affine, best_pts_pair_affine)

cv2.imwrite(os.path.join('output', 'ps4-3-b-1.png'), best_match_img_sim)
cv2.imwrite(os.path.join('output', 'ps4-3-c-1.png'), best_match_img_affine)
'''

rows,cols = img_sim_b.shape
trans_sqmat_sim = np.append(best_trans_sim, np.array([[0, 0, 1]]), axis = 0)
trans_inverse_sim =np.linalg.inv(trans_sqmat_sim)
unwarpped_sim_img = cv2.warpAffine(img_sim_b, trans_inverse_sim[0:2,0:3],(cols, rows))

trans_sqmat_affine = np.append(best_trans_affine, np.array([[0, 0, 1]]), axis = 0)
trans_inverse_affine =np.linalg.inv(trans_sqmat_affine)
unwarpped_affine_img = cv2.warpAffine(img_sim_b, trans_inverse_affine[0:2,0:3],(cols, rows))

#cv2.imwrite(os.path.join('output', 'ps4-3-d-1.png'), unwarpped_sim_img)
#cv2.imwrite(os.path.join('output', 'ps4-3-e-1.png'), unwarpped_affine_img)

blended_img_sim = util.blend_image(img_sim_a, unwarpped_sim_img)
blended_img_affine = util.blend_image(img_sim_a, unwarpped_affine_img)

cv2.imwrite(os.path.join('output', 'ps4-3-d-2.png'), blended_img_sim)
cv2.imwrite(os.path.join('output', 'ps4-3-e-2.png'), blended_img_affine)