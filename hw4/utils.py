import cv2
import numpy as np
import os
import random

from tqdm import trange
from icecream import ic


def match_keypoints(keypoints_l, keypoints_r, descriptors_l, descriptors_r, ratio):
    pair_points = []
    for i in trange(len(keypoints_l)):
        dists = np.linalg.norm(descriptors_r - descriptors_l[i], axis=1)
        parted_idx = np.argpartition(dists, 2)[:2]
        if dists[parted_idx[0]] < ratio * dists[parted_idx[1]]:
            pair_points.append([keypoints_l[i].pt, keypoints_r[parted_idx[0]].pt])
    return pair_points


def feature_matching(img1, img2, ratio=0.75, output_path="output"):
    sift = cv2.SIFT_create()  # type: ignore
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    ic(des1.shape, des2.shape)

    feat1 = cv2.drawKeypoints(img1, kp1, img1)
    feat2 = cv2.drawKeypoints(img2, kp2, img2)

    cv2.imwrite(
        os.path.join(output_path, "feature_matching.jpg"),
        np.hstack((feat1, feat2)),
    )

    pair_points = match_keypoints(kp1, kp2, des1, des2, ratio)

    ic(len(pair_points))
    return pair_points


def normalize_points(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    scale = np.sqrt(2) / np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))

    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    normalized_points = (T @ np.column_stack((points, np.ones(len(points)))).T).T
    return normalized_points[:, :2], T


def compute_fundamental_matrix(pts_src, pts_dst):
    pts_src_norm, T1 = normalize_points(pts_src)
    pts_dst_norm, T2 = normalize_points(pts_dst)

    A = []
    for (x1, y1), (x2, y2) in zip(pts_src_norm, pts_dst_norm):
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    # print(U.shape, S.shape, Vt.shape, F.shape)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    F = T2.T @ F @ T1

    return F / F[2, 2]


def ransac(matches_pos, threshold=5, max_iterations=1000):
    max_inliers = 0
    best_F = None
    for _ in trange(max_iterations):
        sample_indices = random.sample(range(len(matches_pos)), 8)
        src_sample = np.array([matches_pos[i][1] for i in sample_indices])
        dst_sample = np.array([matches_pos[i][0] for i in sample_indices])

        F = compute_fundamental_matrix(src_sample, dst_sample)

        inlier_count = 0
        for match in matches_pos:
            src_point = np.array([*match[1], 1])
            dst_point = np.array([*match[0], 1])
            line = F @ src_point
            distance = np.abs(dst_point @ line) / np.sqrt(line[0] ** 2 + line[1] ** 2)
            if distance < threshold:
                inlier_count += 1
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_F = F
    print("Number of inliers:", max_inliers)
    return best_F
