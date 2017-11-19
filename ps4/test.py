import numpy as np
import math
import os
from scipy import signal
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
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
    points, descriptors = sift.compute(img, key_pts_list)

    return points, descriptors

img1 = cv2.imread(os.path.join('input', 'simA.jpg'),0)
img2 = cv2.imread(os.path.join('input', 'simB.jpg'),0)
points1, descriptors1 = run_main('simA.jpg')
points2, descriptors2 = run_main('simB.jpg')

matches, match_img = util.get_matches(img1, img2, points1, points2, descriptors1, descriptors2)

#best_trans, best_pts_pair = util.get_ransac_affine_trans(points1, points2, matches, 50, 0.99)
best_trans, best_pts_pair = util.get_ransac_sim_trans(points1, points2, matches, 50, 0.99)
#best_trans, best_pts_pair = util.get_ransac_translation(points1, points2, matches, 100, 0.99)

best_match_img = cv2.cvtColor(np.concatenate((img1, img2), axis=1),cv2.COLOR_GRAY2RGB)

pt1 = (int(img1.shape[1]/2), int(img2.shape[0]/2))
pt1_vec = np.array([pt1[0], pt1[1], 1])
pt2_vec = np.dot(best_trans, pt1_vec.T)
pt2 = (int(pt2_vec[0] + img1.shape[1]), int(pt2_vec[1]))
#cv2.line(best_match_img, pt1, pt2, (0, 255, 0), 5)

rows,cols = img2.shape
#util.draw_pair(best_match_img, best_pts_pair)

trans_sqmat = np.append(best_trans, np.array([[0, 0, 1]]), axis = 0)
best_trans_inverse =np.linalg.inv(trans_sqmat)
dst = cv2.warpAffine(img2,best_trans_inverse[0:2,0:3],(cols, rows))
blended_img = util.blend_image(img1, dst)
'''
cv2.imshow('image', stacked_g_img)
cv2.imshow('image1', angle_map_norm)
cv2.imshow('image2', r_norm)

'''

cv2.imshow('image3', blended_img)

cv2.waitKey(0)



def test_gaussian():
    h = util.gaussian2D((3, 3), 0.75)
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    h_x = signal.convolve2d(h, prewitt_x)
    h_y = signal.convolve2d(h, prewitt_y)
    print h_x.shape

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')

    # Make data.

    X, Y = np.meshgrid(np.linspace(-1, 1, len(h_x[0, :])), np.linspace(-1, 1, len(h_x[:, 0])))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, h_x)

    X, Y = np.meshgrid(np.linspace(-1, 1, len(h[0, :])), np.linspace(-1, 1, len(h[:, 0])))
    # Plot the surface.
    surf1 = ax1.plot_surface(X, Y, h)

    plt.show()
    return