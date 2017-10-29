import numpy as np

def svd_M_solver(pts_2d, pts_3d):
    #  M = np.zeros((12,1), dtype=np.float32)
    num_pts = pts_2d.shape[0]
    A = np.zeros((2*num_pts,12), dtype=np.float32)
    b = np.zeros(2*num_pts, dtype=np.float32)
    x = pts_2d[:,0]
    y = pts_2d[:,1]
    X = pts_3d[:,0]
    Y = pts_3d[:,1]
    Z = pts_3d[:,2]
    zeros = np.zeros(num_pts)
    ones = np.ones(num_pts)
    A[::2,:]   = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -x*X, -x*Y, -x*Z, -x))
    A[1::2,:] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones, -y*X, -y*Y, -y*Z, -y))
    _,_,V = np.linalg.svd(A, full_matrices=True)
    M = V.T[:,-1]
    M = M.reshape((3,4))
    return M

def svd_F_solver(pts_a, pts_b):
    num_pts = pts_a.shape[0]
    ua = pts_a[:,0]
    va = pts_a[:,1]
    ub = pts_b[:,0]
    vb = pts_b[:,1]
    ones = np.ones(num_pts)
    A = np.column_stack((ua*ub, va*ub, ub, ua*vb, va*vb, vb, ua, va, ones))
    _,_,V = np.linalg.svd(A, full_matrices=True)
    F = V.T[:,-1]
    F = F.reshape((3,3))
    return F