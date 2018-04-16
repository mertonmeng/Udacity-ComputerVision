import numpy as np
import cv2
import os
import math

def calculate_moment(img, p, q):
    mesh = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    y_mesh = mesh[0].astype(float)
    x_mesh = mesh[1].astype(float)
    img_sum = np.sum(img)
    M00 = img_sum
    M01 = np.sum(img * y_mesh)
    M10 = np.sum(img * x_mesh)
    x_avg = M10/M00
    y_avg = M01/M00
    mu = np.sum(((x_mesh - x_avg) ** p) * ((y_mesh - y_avg) ** q) * img)
    eta = mu/(img_sum ** (1.0 + (p + q)/2.0))

    return mu, eta

def get_dist(vec1, vec2):
    dist = (np.sum((vec1 - vec2) ** 2))**0.5
    return dist