import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random


def stitch(img_pair, ratio=0.75):
    img_l, img_r = img_pair
    (hl, wl) = img_l.shape[:2]
    (hr, wr) = img_r.shape[:2]

    print("Step 1 - SIFT keypoint detection and descriptor computation...")
    sift = cv2.SIFT_create()
    keypoints_l, descriptors_l = sift.detectAndCompute(img_l, None)
    keypoints_r, descriptors_r = sift.detectAndCompute(img_r, None)

    print("Step 2 - Keypoint matching with Lowe's ratio test...")
    matches_pos = matchKeyPoints(
        keypoints_l, keypoints_r, descriptors_l, descriptors_r, ratio
    )
    print("Number of matching points:", len(matches_pos))

    print("Step 3 - RANSAC algorithm to find the best homography...")
    HomoMat = ransacHomography(matches_pos)

    print("Step 4 - Create panoramic image...")
    warp_img = warp([img_l, img_r], HomoMat)

    return warp_img


def matchKeyPoints(keypoints_l, keypoints_r, descriptors_l, descriptors_r, ratio):
    pair_points = []
    for i in range(len(descriptors_l)):
        distances = [np.linalg.norm(descriptors_l[i] - d) for d in descriptors_r]
        sorted_indices = np.argsort(distances)
        min_d, second_min_d = distances[sorted_indices[0]], distances[sorted_indices[1]]

        if min_d < ratio * second_min_d:
            pair_points.append(
                [
                    (int(keypoints_l[i].pt[0]), int(keypoints_l[i].pt[1])),
                    (
                        int(keypoints_r[sorted_indices[0]].pt[0]),
                        int(keypoints_r[sorted_indices[0]].pt[1]),
                    ),
                ]
            )

    return pair_points


def findHomographyMatrix(src, dst):
    A = []
    for i in range(len(src)):
        x, y = src[i][0], src[i][1]
        x_prime, y_prime = dst[i][0], dst[i][1]
        A.extend(
            [
                [-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime],
                [0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime],
            ]
        )

    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def ransacHomography(matches_pos, threshold=5.0, max_iterations=8000):
    max_inliers = 0
    best_H = None

    for _ in range(max_iterations):
        sample_indices = random.sample(range(len(matches_pos)), 4)
        src_sample = np.array([matches_pos[i][1] for i in sample_indices])
        dst_sample = np.array([matches_pos[i][0] for i in sample_indices])

        H = findHomographyMatrix(src_sample, dst_sample)
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


def warp(imgs, HomoMat):

    img_left, img_right = imgs
    hl, wl = img_left.shape[:2]
    hr, wr = img_right.shape[:2]
    stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="int")

    # Compute the inverse of the homography matrix for coordinate mapping
    inv_H = np.linalg.inv(HomoMat)

    # Map pixels from the right image to the stitched image using the inverse homography
    for i in range(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            coor = np.array([j, i, 1])
            img_right_coor = inv_H @ coor
            img_right_coor /= img_right_coor[2]

            x, y = int(round(img_right_coor[1])), int(
                round(img_right_coor[0])
            )  # Map coordinates

            # Skip pixels outside the boundaries of the right image
            if 0 <= x < hr and 0 <= y < wr:
                stitch_img[i, j] = img_right[x, y]

    # Blend the left and transformed right image
    stitch_img = linear_blending([img_left, stitch_img])
    return remove_black_border(stitch_img)


def linear_blending(imgs):

    img_left, img_right = imgs
    hl, wl = img_left.shape[:2]
    hr, wr = img_right.shape[:2]

    # Ensure img_right is resized to match img_left's width for blending
    img_right_resized = np.zeros((max(hl, hr), wl + wr, 3), dtype=img_right.dtype)
    img_right_resized[:hr, :wr] = img_right

    img_left_mask = (img_left.sum(axis=2) > 0).astype(int)
    img_right_mask = (img_right_resized.sum(axis=2) > 0).astype(int)
    overlap_mask = np.zeros_like(img_right_mask)
    overlap_mask[:hl, :wl] = img_left_mask & img_right_mask[:hl, :wl]

    alpha_mask = np.zeros((max(hl, hr), wl + wr))
    for i in range(max(hl, hr)):
        overlap_indices = np.where(overlap_mask[i] == 1)[0]
        if overlap_indices.size > 1:
            min_idx, max_idx = overlap_indices[0], overlap_indices[-1]
            alpha_mask[i, min_idx : max_idx + 1] = 1 - np.linspace(
                0, 1, max_idx - min_idx + 1
            )

    blended_img = img_right_resized.copy()
    blended_img[:hl, :wl] = img_left.copy()

    for i in range(max(hl, hr)):
        for j in range(wl + wr):
            if overlap_mask[i, j]:
                blended_img[i, j] = (
                    alpha_mask[i, j] * img_left[i, j]
                    + (1 - alpha_mask[i, j]) * img_right_resized[i, j]
                )

    return blended_img


def remove_black_border(img):

    non_black_cols = np.where(img.max(axis=0) > 0)[0]
    non_black_rows = np.where(img.max(axis=1) > 0)[0]

    if non_black_cols.size and non_black_rows.size:
        return img[
            non_black_rows[0] : non_black_rows[-1] + 1,
            non_black_cols[0] : non_black_cols[-1] + 1,
        ]

    return img


if __name__ == "__main__":
    data_folder = "data/"
    file_names = os.listdir(data_folder)
    fileNameList = [
        (file_names[i], file_names[i + 1]) for i in range(0, len(file_names), 2)
    ]

    for fname1, fname2 in fileNameList:
        img_left = cv2.imread(os.path.join(data_folder, fname1))
        img_right = cv2.imread(os.path.join(data_folder, fname2))

        warp_img = stitch([img_left, img_right])

        plt.figure()
        plt.title("Stitched Image")

        # Ensure the image is in uint8 format for proper display
        if warp_img.dtype != np.uint8:
            warp_img = np.clip(warp_img, 0, 255).astype(np.uint8)

        plt.imshow(cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB))

        save_path = os.path.join("result", fname1)
        cv2.imwrite(save_path, warp_img)
