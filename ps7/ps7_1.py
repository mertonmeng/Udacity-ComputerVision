import numpy as np
import cv2
import os
import math

cap = cv2.VideoCapture('input\\PS7A1P1T1.avi')
t = 0
ret, frame = cap.read()
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
tau = 50
mhi_img = np.zeros(prev_frame.shape, dtype=float)
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

    frame_diff = cv2.normalize(frame_diff, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('frame', bool_frame)
    if t == tau:
        mhi_img = cv2.normalize(mhi_img, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite(os.path.join('output', 'ps7-1-b-1.png'), mhi_img * 255)
        cv2.imshow('MHI', mhi_img)
    '''
    if t == 10:
        cv2.imwrite(os.path.join('output', 'ps7-1-a-1.png'), bool_frame*255)
    if t == 20:
        cv2.imwrite(os.path.join('output', 'ps7-1-a-2.png'), bool_frame*255)
    if t == 30:
        cv2.imwrite(os.path.join('output', 'ps7-1-a-3.png'), bool_frame*255)
    '''


    prev_frame = frame_gray
    t += 1

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()