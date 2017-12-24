import numpy as np
import cv2
import os
import pyramids

original_img = cv2.imread(os.path.join('input\\DataSeq1', 'yos_img_01.jpg'),0)
pyramids_lst = pyramids.gaussian_pyramids_gen(original_img, 4)

combined_img = pyramids_lst[0]
for i in range(1,len(pyramids_lst)):
    pad_img = np.zeros(pyramids_lst[0].shape, dtype=np.uint8)
    cur_layer = pyramids_lst[i]
    pad_img[0:cur_layer.shape[0], 0:cur_layer.shape[1]] = cur_layer
    combined_img = np.concatenate((combined_img, pad_img), axis=1)

cv2.imwrite(os.path.join('output', 'ps5-2-a-1.png'), combined_img)


lap_pyramids = pyramids.laplacian_pyramids_gen(pyramids_lst)
lap_combined_img = lap_pyramids[0]

for i in range(1,len(lap_pyramids)):
    pad_img = np.zeros(lap_pyramids[0].shape, dtype=np.uint8)
    cur_layer = lap_pyramids[i]
    pad_img[0:cur_layer.shape[0], 0:cur_layer.shape[1]] = cur_layer
    lap_combined_img = np.concatenate((lap_combined_img, pad_img), axis=1)

cv2.imwrite(os.path.join('output', 'ps5-2-b-1.png'), lap_combined_img)
