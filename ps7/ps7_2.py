import numpy as np
import cv2
import os
import math
import algo

pq_pair = np.array([[2,0],[0,2],[1,2],[2,1],[2,2],[3,0],[0,3]])

def get_MEI_MHI(path):
    cap = cv2.VideoCapture(path)
    t = 0
    ret, frame = cap.read()
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    tau = 50
    mhi_img = np.zeros(prev_frame.shape, dtype=float)
    mei_img = np.zeros(prev_frame.shape, dtype=float)
    tau_mask = np.zeros(prev_frame.shape) + tau

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (31,31),0)
        frame_diff = np.abs(np.subtract(frame_gray.astype(np.float),prev_frame.astype(np.float)))

        bool_frame = (frame_diff >= 20).astype(np.uint8)
        bool_frame = cv2.morphologyEx(bool_frame, cv2.MORPH_OPEN, kernel)
        bool_frame = cv2.normalize(bool_frame, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        non_active_mask = np.clip(mhi_img - 1.0, 0, 255) * (bool_frame == 0.0)
        mhi_img = tau_mask * bool_frame + non_active_mask

        #frame_diff = cv2.normalize(frame_diff, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #cv2.imshow('frame', bool_frame)
        if t == tau:
            mhi_img = cv2.normalize(mhi_img, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            mei_img = (mhi_img > 0).astype(float)

            break

        prev_frame = frame_gray
        t += 1

    cap.release()
    cv2.destroyAllWindows()
    return mhi_img, mei_img

def get_vector(mhi_img, mei_img, pq_pair):
    mu_vec = []
    eta_vec = []
    for i in range(0, pq_pair.shape[0]):
        mu_mhi, eta_mhi = algo.calculate_moment(mhi_img, pq_pair[i, 0], pq_pair[i, 1])
        mu_mei, eta_mei = algo.calculate_moment(mei_img, pq_pair[i, 0], pq_pair[i, 1])
        mu_vec.append(mu_mhi)
        mu_vec.append(mu_mei)
        eta_vec.append(eta_mhi)
        eta_vec.append(eta_mei)

    mu_vec = np.array(mu_vec)
    eta_vec = np.array(eta_vec)
    return mu_vec, eta_vec

def generate_confusion_matrix(vec_list1, vec_list2, video_list, csv_path):
    confusion_mat_mu = np.zeros((video_list.shape[0], vec_list2.shape[0]))
    confusion_mat_eta = np.zeros(confusion_mat_mu.shape)
    for i in range(0, video_list.shape[0]):
        for k in range(0, video_list.shape[1]):
            min_eta_dist = float('inf')
            min_mu_dist = float('inf')
            min_eta_idx = 0
            min_mu_idx = 0
            for j in range(0, vec_list1.shape[0]):
                mu_vec_train = vec_list1[j]
                eta_vec_train = vec_list2[j]
                path = video_list[i][k]
                mhi_img, mei_img = get_MEI_MHI(path)
                mu_vec_test,eta_vec_test = get_vector(mhi_img, mei_img, pq_pair)
                mu_dist = algo.get_dist(mu_vec_test, mu_vec_train)
                eta_dist = algo.get_dist(eta_vec_test, eta_vec_train)
                if mu_dist < min_mu_dist:
                    min_mu_dist = mu_dist
                    min_mu_idx = j
                if eta_dist < min_eta_dist:
                    min_eta_dist = eta_dist
                    min_eta_idx = j

            confusion_mat_mu[i][min_mu_idx] += 1
            confusion_mat_eta[i][min_eta_idx] += 1
        confusion_mat_mu[i] = confusion_mat_mu[i]/np.sum(confusion_mat_mu[i])
        confusion_mat_eta[i] = confusion_mat_eta[i] / np.sum(confusion_mat_eta[i])

    np.savetxt(csv_path + '_mu.csv', confusion_mat_mu, fmt='%.2f', delimiter=',', newline='\n')
    np.savetxt(csv_path + '_eta.csv', confusion_mat_eta, fmt='%.2f', delimiter=',', newline='\n')
    return

mhi_img_a1, mei_img_a1 = get_MEI_MHI('input\\PS7A1P1T1.avi')
mhi_img_a2, mei_img_a2 = get_MEI_MHI('input\\PS7A2P1T1.avi')
mhi_img_a3, mei_img_a3 = get_MEI_MHI('input\\PS7A3P1T1.avi')

mu_vec_a1, eta_vec_a1 = get_vector(mhi_img_a1, mei_img_a1, pq_pair)
mu_vec_a2, eta_vec_a2 = get_vector(mhi_img_a2, mei_img_a2, pq_pair)
mu_vec_a3, eta_vec_a3 = get_vector(mhi_img_a3, mei_img_a3, pq_pair)

mu_vec_list = np.array([mu_vec_a1, mu_vec_a2, mu_vec_a3])
eta_vec_list = np.array([eta_vec_a1, eta_vec_a2, eta_vec_a3])

video_list_p1 = np.array([['input\\PS7A1P1T1.avi', 'input\\PS7A1P1T2.avi', 'input\\PS7A1P1T3.avi'],
                   ['input\\PS7A2P1T1.avi', 'input\\PS7A2P1T2.avi', 'input\\PS7A2P1T3.avi'],
                   ['input\\PS7A3P1T1.avi', 'input\\PS7A3P1T2.avi', 'input\\PS7A3P1T3.avi']])

#generate_confusion_matrix(mu_vec_list, eta_vec_list, video_list_p1, 'output\\ps7-2-a')

video_list_all = np.array([['input\\PS7A1P1T1.avi', 'input\\PS7A1P1T2.avi', 'input\\PS7A1P1T3.avi',
                           'input\\PS7A1P2T1.avi', 'input\\PS7A1P2T2.avi', 'input\\PS7A1P2T3.avi',
                           'input\\PS7A1P3T1.avi', 'input\\PS7A1P3T2.avi', 'input\\PS7A1P3T3.avi'],
                   ['input\\PS7A2P1T1.avi', 'input\\PS7A2P1T2.avi', 'input\\PS7A2P1T3.avi',
                    'input\\PS7A2P2T1.avi', 'input\\PS7A2P2T2.avi', 'input\\PS7A2P2T3.avi',
                    'input\\PS7A2P3T1.avi', 'input\\PS7A2P3T2.avi', 'input\\PS7A2P3T3.avi'],
                   ['input\\PS7A3P1T1.avi', 'input\\PS7A3P1T2.avi', 'input\\PS7A3P1T3.avi',
                    'input\\PS7A3P2T1.avi', 'input\\PS7A3P2T2.avi', 'input\\PS7A3P2T3.avi',
                    'input\\PS7A3P3T1.avi', 'input\\PS7A3P1T2.avi', 'input\\PS7A3P3T3.avi']])

generate_confusion_matrix(mu_vec_list, eta_vec_list, video_list_all, 'output\\ps7-2-b')

cv2.imshow('MHI', mhi_img_a2)
cv2.imshow('MEI', mei_img_a2)
cv2.waitKey(0)

