import cv2
import numpy as np
import os
import random

from tqdm import trange
from icecream import ic

from config import OUTPUT_PATH


def match_keypoints(keypoints_l, keypoints_r, descriptors_l, descriptors_r, ratio):
    pair_points = []
    for i in trange(len(keypoints_l)):
        dists = np.linalg.norm(descriptors_r - descriptors_l[i], axis=1)
        parted_idx = np.argpartition(dists, 2)[:2]
        if dists[parted_idx[0]] < ratio * dists[parted_idx[1]]:
            pair_points.append([keypoints_l[i].pt, keypoints_r[parted_idx[0]].pt])
    return pair_points


def feature_matching(img1, img2, ratio=0.75):
    sift = cv2.SIFT_create()  # type: ignore
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    ic(des1.shape, des2.shape)

    feat1 = cv2.drawKeypoints(img1, kp1, img1)
    feat2 = cv2.drawKeypoints(img2, kp2, img2)
    cv2.imwrite(
        os.path.join(OUTPUT_PATH, "feature_matching.jpg"),
        np.hstack((feat1, feat2)),
    )

    pair_points = match_keypoints(kp1, kp2, des1, des2, ratio)

    ic(len(pair_points))
    return pair_points


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


def ransac(matches_pos, threshold=5.0, max_iterations=1000):
    max_inliers = 0
    best_H = None
    for _ in trange(max_iterations):
        sample_indices = random.sample(range(len(matches_pos)), 8)
        src_sample = np.array([matches_pos[i][1] for i in sample_indices])
        dst_sample = np.array([matches_pos[i][0] for i in sample_indices])

        H = compute_homography(src_sample, dst_sample)
        inlier_count = 0

        for match in matches_pos:
            src_point = np.array([*match[1], 1])
            projected_point = H @ src_point
            projected_point /= projected_point[2]
            dst_point = np.array([*match[0]])
            if np.linalg.norm(projected_point[:2] - dst_point) < threshold:
                inlier_count += 1
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_H = H

    print("Number of inliers:", max_inliers)
    return best_H
