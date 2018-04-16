import numpy as np
import cv2
import algo
import os
import math

cap = cv2.VideoCapture('input\\pres_debate.avi')

x0 = 160
y0 = 88
w = 20
h = 20

t = 0
template = np.zeros((h, w))

num_sample = 100
sample_set = []

center_x = float(x0 + w / 2)
center_y = float(y0 + h / 2)

alpha = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (int(0.5 * frame.shape[1]), int(0.5 * frame.shape[0])), interpolation=cv2.INTER_CUBIC)

    if t == 0:
        template = frame[y0: y0 + h, x0: x0 + w, :]
        padded = np.zeros(((frame.shape[0] + h), (frame.shape[1] + w), 3))
        padded[h / 2: -h / 2, w / 2: -w / 2, :] = frame
        target = np.zeros(template.shape)
        dist_map = np.zeros((frame.shape[0], frame.shape[1]))

        for x in range(0, frame.shape[1]):
            for y in range(0, frame.shape[0]):
                target = padded[y: y + h, x: x + w, :]
                hist_dist = algo.histogram_distance(template, target)
                dist_map[y, x] = hist_dist

        dist_map = cv2.normalize(dist_map, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow('dist_map', dist_map)

    t += 1

cv2.waitKey()
cap.release()
cv2.destroyAllWindows()