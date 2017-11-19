import numpy as np
import sys
import math
import cv2
import copy
from scipy import signal

def gaussian2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def get_gradient_img(img, kernel_size, sigma):
    h = gaussian2D(kernel_size, sigma)
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    h_x = signal.convolve2d(h, prewitt_x, mode='valid', boundary='symm')
    h_y = signal.convolve2d(h, prewitt_y, mode='valid', boundary='symm')

    gx = signal.convolve2d(img, h_x, mode='valid', boundary='symm')
    gy = signal.convolve2d(img, h_y, mode='valid', boundary='symm')

    gx_norm = np.zeros(gx.shape)
    gy_norm = np.zeros(gy.shape)

    if (gx.max() - gx.min()) > 0:
        gx_norm = (gx + np.abs(gx.min())) / (gx.max() - gx.min())
    else:
        gx_norm = gx

    if (gy.max() - gy.min()) > 0:
        gy_norm = (gy + np.abs(gy.min())) / (gy.max() - gy.min())
    else:
        gy_norm = gx

    return gx, gy, gx_norm, gy_norm

def get_harris_value(gx, gy, kernel_size):
    alpha = 0.05
    r_mat = np.zeros(gx.shape)
    r_norm_mat = np.zeros(gx.shape)
    w = gaussian2D(kernel_size, kernel_size[0]/5)
    half_wid = int((w.shape[0])/2)

    for i in range(half_wid, len(gx[:,0]) - half_wid):
        for j in range(half_wid, len(gx[0,:]) - half_wid):
            gx_patch = gx[i - half_wid: i + half_wid + 1, j - half_wid: j + half_wid + 1]
            gy_patch = gy[i - half_wid: i + half_wid + 1, j - half_wid: j + half_wid + 1]
            elem1 = np.sum(w * gx_patch * gx_patch)
            elem23 = np.sum(w * gx_patch * gy_patch)
            elem4 = np.sum(w * gy_patch * gy_patch)
            m = np.array([[elem1, elem23],[elem23, elem4]])
            r_mat[i,j] = np.linalg.det(m) - alpha * (np.trace(m) ** 2)

    if (r_mat.max() - r_mat.min()) > 0:
        r_norm_mat = (r_mat + np.abs(r_mat.min())) / (r_mat.max() - r_mat.min())
    else:
        r_norm_mat = r_mat

    return r_mat, r_norm_mat

def find_corner_point(r, half_wid, threshold):
    pt_list = []

    thresh_val = r.max()*threshold

    for i in range(half_wid, len(r[:,0]) - half_wid):
        for j in range(half_wid, len(r[0,:]) - half_wid):
            r_patch = r[i - half_wid: i + half_wid + 1, j - half_wid: j + half_wid + 1]
            max_val = np.max(r_patch)
            if (max_val == r_patch[half_wid, half_wid]) and (max_val > thresh_val) and (max_val > 0):
                pt_list.append([j, i])

    return np.array(pt_list)

def draw_points(img, pt_arr):
    img_with_graphics = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    for pt in pt_arr:
        pt_tup = tuple(pt)
        cv2.circle(img_with_graphics, pt_tup, 4, (0,255,0))

    return img_with_graphics

def draw_interest_points(img, key_pts):
    img_with_graphics = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawKeypoints(img_with_graphics, key_pts, img_with_graphics, (0, 255, 0))
    for key_pt in key_pts:
        strat_pt = (int(key_pt.pt[0]), int(key_pt.pt[1]))
        end_pt = (int(key_pt.pt[0] + 10*math.cos(math.radians(key_pt.angle))),
                  int(key_pt.pt[1] + 10*math.sin(math.radians(key_pt.angle))))
        cv2.line(img_with_graphics,strat_pt, end_pt, (0, 255, 0))

    return img_with_graphics

def get_angle_map(gx, gy):
    angle_map = np.arctan2(gy,gx)
    angle_map_norm = cv2.normalize(angle_map, angle_map, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return angle_map, angle_map_norm

def get_key_points(corner_pts, angle_map):
    key_points_list = []
    for corner_pt in corner_pts:
        pt_x = corner_pt[0]
        pt_y = corner_pt[1]
        pt_angle = math.degrees(angle_map[pt_y, pt_x])
        key_pt = cv2.KeyPoint(x = pt_x, y = pt_y, _size = 20, _angle=pt_angle, _octave=0)
        key_points_list.append(key_pt)

    return key_points_list

def get_matches(img1, img2, key_pts1, key_pts2, descriptors1, descriptors2):
    bfm = cv2.BFMatcher()
    matches = bfm.match(descriptors1, descriptors2)

    match_img = cv2.cvtColor(np.concatenate((img1, img2), axis=1),cv2.COLOR_GRAY2RGB)

    for match in matches:
        pt_a = key_pts1[match.queryIdx].pt
        pt_b = key_pts2[match.trainIdx].pt
        pt_a = (int(pt_a[0]), int(pt_a[1]))
        pt_b = (int(pt_b[0]) + img1.shape[1], int(pt_b[1]))
        cv2.circle(match_img, pt_a, 4, (0, 255, 0))
        cv2.circle(match_img, pt_b, 4, (0, 255, 0))
        cv2.line(match_img, pt_a, pt_b, (0, 255, 0))

    return matches, match_img

def get_ransac_translation(keypoints1, keypoints2, matches, dist_thresh, p):
    N = sys.maxint
    sample_count = 0
    trans_list = []
    e = 1
    best_pair_list = []

    for match in matches:
        key_pt1 = keypoints1[match.queryIdx]
        key_pt2 = keypoints2[match.trainIdx]
        t_x = key_pt1.pt[0] - key_pt2.pt[0]
        t_y = key_pt1.pt[1] - key_pt2.pt[1]
        trans_list.append([t_x, t_y])

    max_inlier_num = 0
    best_trans = []

    while (N > sample_count) and (sample_count < len(matches)):
        match = matches[sample_count]
        key_pt1 = keypoints1[match.queryIdx]
        key_pt2 = keypoints2[match.trainIdx]
        t_x = key_pt1.pt[0] - key_pt2.pt[0]
        t_y = key_pt1.pt[1] - key_pt2.pt[1]

        sum_tx = 0
        sum_ty = 0
        inlier_num = 0
        key_pts_pair_list = []

        for match in matches:
            key_pt1_temp = keypoints1[match.queryIdx]
            key_pt2_temp = keypoints2[match.trainIdx]
            t_x_temp = key_pt1_temp.pt[0] - key_pt2_temp.pt[0]
            t_y_temp = key_pt1_temp.pt[1] - key_pt2_temp.pt[1]
            dist = math.sqrt((t_x - t_x_temp)**2 + (t_y - t_y_temp)**2)
            if dist < dist_thresh:
                key_pts_pair_list.append([key_pt1_temp.pt, key_pt2_temp.pt])
                inlier_num += 1
                sum_tx += t_x_temp
                sum_ty += t_y_temp

        if inlier_num > max_inlier_num:
            best_trans = [sum_tx/inlier_num, sum_ty/inlier_num]
            max_inlier_num = inlier_num
            best_pair_list = key_pts_pair_list

        e0 = 1 - inlier_num/float(len(trans_list))
        if e0 < e:
            e = e0
            N = math.log10(1 - p) / math.log10(e)
        sample_count += 1

    return best_trans, best_pair_list

def get_ransac_sim_trans(keypoints1, keypoints2, matches, dist_thresh, p):
    N = sys.maxint
    sample_count = 0
    best_pair_list = []
    e = 1

    max_inlier_num = 0
    best_sim_trans = np.zeros((2,3))

    while (N > sample_count) and (2 * sample_count + 1 < len(matches)):
        match1 = matches[2 * sample_count]
        match2 = matches[2 * sample_count + 1]

        M = solve_sim_mat(match1, match2, keypoints1, keypoints2)
        key_pts_pair_list = []

        inlier_num = 0
        for match in matches:
            key_pt1 = keypoints1[match.queryIdx].pt
            key_pt2 = keypoints2[match.trainIdx].pt
            vec = (np.array([key_pt1[0], key_pt1[1], 1])).T
            vec_proj = np.dot(M, vec)


            dist = math.sqrt((vec_proj[0] - key_pt2[0])**2 + (vec_proj[1] - key_pt2[1])**2)
            if dist < dist_thresh:
                key_pts_pair_list.append([key_pt1, key_pt2])
                inlier_num += 1

        if inlier_num > max_inlier_num:
            best_sim_trans = M
            best_pair_list = key_pts_pair_list
            max_inlier_num = inlier_num

        e0 = 1 - inlier_num/float(len(matches))
        if e0 < e:
            e = e0
            N = math.log10(1 - p) / math.log10(1 - (1 - e)**2)
        sample_count += 1

    return best_sim_trans, best_pair_list

def get_ransac_affine_trans(keypoints1, keypoints2, matches, dist_thresh, p):
    N = sys.maxint
    sample_count = 0
    best_pair_list = []
    e = 1

    max_inlier_num = 0
    best_affine_trans = np.zeros((2,3))

    while (N > sample_count) and (3 * sample_count + 2 < len(matches)):
        match1 = matches[3 * sample_count]
        match2 = matches[3 * sample_count + 1]
        match3 = matches[3 * sample_count + 2]

        M = solve_affine_mat(match1, match2, match3, keypoints1, keypoints2)
        key_pts_pair_list = []

        inlier_num = 0
        for match in matches:
            key_pt1 = keypoints1[match.queryIdx].pt
            key_pt2 = keypoints2[match.trainIdx].pt
            vec = (np.array([key_pt1[0], key_pt1[1], 1])).T
            vec_proj = np.dot(M, vec)


            dist = math.sqrt((vec_proj[0] - key_pt2[0])**2 + (vec_proj[1] - key_pt2[1])**2)
            if dist < dist_thresh:
                key_pts_pair_list.append([key_pt1, key_pt2])
                inlier_num += 1

        if inlier_num > max_inlier_num:
            best_affine_trans = M
            best_pair_list = key_pts_pair_list
            max_inlier_num = inlier_num

        e0 = 1 - inlier_num/float(len(matches))
        if e0 < e:
            e = e0
            N = math.log10(1 - p) / math.log10(1 - (1 - e)**2)
        sample_count += 1

    return best_affine_trans, best_pair_list

def solve_sim_mat(match1, match2, keypoints1, keypoints2):
    key_pt11 = keypoints1[match1.queryIdx]
    key_pt12 = keypoints2[match1.trainIdx]
    key_pt21 = keypoints1[match2.queryIdx]
    key_pt22 = keypoints2[match2.trainIdx]

    vec1 = np.array([key_pt11.pt[0], -key_pt11.pt[1], 1, 0, -key_pt12.pt[0]])
    vec2 = np.array([key_pt11.pt[1], key_pt11.pt[0], 0, 1, -key_pt12.pt[1]])
    vec3 = np.array([key_pt21.pt[0], -key_pt21.pt[1], 1, 0, -key_pt22.pt[0]])
    vec4 = np.array([key_pt21.pt[1], key_pt21.pt[0], 0, 1, -key_pt22.pt[1]])

    vec_list = []
    vec_list.append(vec1)
    vec_list.append(vec2)
    vec_list.append(vec3)
    vec_list.append(vec4)
    A = np.array(vec_list)

    U, s, VTrans = np.linalg.svd(A)
    V = VTrans.T
    MVec = V[:, len(V[0, :]) - 1]
    MVec = MVec / MVec[4]
    M = np.array([[MVec[0], -MVec[1], MVec[2]], [MVec[1], MVec[0], MVec[3]]])

    return M

def solve_affine_mat(match1, match2, match3, keypoints1, keypoints2):
    pt11 = keypoints1[match1.queryIdx].pt
    pt12 = keypoints2[match1.trainIdx].pt
    pt21 = keypoints1[match2.queryIdx].pt
    pt22 = keypoints2[match2.trainIdx].pt
    pt31 = keypoints1[match3.queryIdx].pt
    pt32 = keypoints2[match3.trainIdx].pt

    vec11 = np.array([pt11[0], pt11[1], 1, 0, 0, 0, -pt12[0]])
    vec12 = np.array([0, 0, 0, pt11[0], pt11[1], 1, -pt12[1]])
    vec21 = np.array([pt21[0], pt21[1], 1, 0, 0, 0, -pt22[0]])
    vec22 = np.array([0, 0, 0, pt21[0], pt21[1], 1, -pt22[1]])
    vec31 = np.array([pt31[0], pt31[1], 1, 0, 0, 0, -pt32[0]])
    vec32 = np.array([0, 0, 0, pt31[0], pt31[1], 1, -pt32[1]])

    vec_list = []
    vec_list.append(vec11)
    vec_list.append(vec12)
    vec_list.append(vec21)
    vec_list.append(vec22)
    vec_list.append(vec31)
    vec_list.append(vec32)
    A = np.array(vec_list)

    U, s, VTrans = np.linalg.svd(A)
    V = VTrans.T
    MVec = V[:, len(V[0, :]) - 1]
    MVec = MVec / MVec[6]
    M = np.array([[MVec[0], MVec[1], MVec[2]], [MVec[3], MVec[4], MVec[5]]])

    return M

def draw_pair(img, pair_list):

    for pair in pair_list:
        pt_a = pair[0]
        pt_b = pair[1]
        pt_a = (int(pt_a[0]), int(pt_a[1]))
        pt_b = (int(pt_b[0]) + img.shape[1]/2, int(pt_b[1]))
        cv2.circle(img, pt_a, 4, (0, 255, 0))
        cv2.circle(img, pt_b, 4, (0, 255, 0))
        cv2.line(img, pt_a, pt_b, (0, 255, 0))

    return

def blend_image(img1, img2):
    pseudo_color_img = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    pseudo_color_img[:, :, 1] = img1
    pseudo_color_img[:, :, 2] = img2
    blended_img = cv2.cvtColor(pseudo_color_img, cv2.COLOR_BGR2GRAY)
    return blended_img