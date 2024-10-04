import numpy as np


def compute_homography(pts_src, pts_dst):
    A = []
    for i in range(len(pts_src)):
        x, y = pts_src[i]
        u, v = pts_dst[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)
    assert A.shape == (len(pts_src) * 2, 9)
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]


def compute_v(H, i, j):
    return np.array(
        [
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j],
        ]
    )


def compute_intrinsic_matrix(Hs):
    A = []
    for H in Hs:
        A.append(compute_v(H, 0, 1))
        A.append(compute_v(H, 0, 0) - compute_v(H, 1, 1))
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    v = V[-1]

    B = np.array(
        [
            [v[0], v[1], v[3]],
            [v[1], v[2], v[4]],
            [v[3], v[4], v[5]],
        ]
    )

    eigvals, eigvecs = np.linalg.eigh(B)
    if np.any(eigvals <= 0):
        eigvals[eigvals <= 0] = np.finfo(float).eps
        B = eigvecs @ np.diag(eigvals) @ eigvecs.T

    K = np.linalg.cholesky(B)
    K = np.linalg.inv(K).T
    return K / K[2, 2]


def compute_extrinsic_matrix(K, H):
    K_inv = np.linalg.inv(K)
    r1, r2, t = np.dot(K_inv, H).T
    t /= (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    r1 /= np.linalg.norm(r1)
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R = np.array([r1, r2, r3]).T
    return np.concatenate((R, t[:, None]), axis=1)
