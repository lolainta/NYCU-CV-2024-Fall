import numpy as np
from tqdm import trange
import cv2
import os
import random

# FOLDER_PATH = "my_data/"
# FOLDER_PATH = "my_data/4k"
FOLDER_PATH = "data"
SEED = 114514

random.seed(SEED)
np.random.seed(SEED)


def stitch(img_pair, fname1, fname2, ratio=0.75):
    img_l, img_r = img_pair
    print("Step 1 - SIFT keypoint detection and descriptor computation...")
    sift = cv2.SIFT_create()  # type: ignore
    keypoints_l, descriptors_l = sift.detectAndCompute(img_l, None)
    keypoints_r, descriptors_r = sift.detectAndCompute(img_r, None)
    print("Number of keypoints in the left image:", len(keypoints_l))
    print("Number of keypoints in the right image:", len(keypoints_r))

    cv2.imwrite(
        f"output/keypoints/{fname1}", cv2.drawKeypoints(img_l, keypoints_l, None)  # type: ignore
    )
    cv2.imwrite(
        f"output/keypoints/{fname2}", cv2.drawKeypoints(img_r, keypoints_r, None)  # type: ignore
    )

    print("Step 2 - Keypoint matching with Lowe's ratio test...")
    matches_pos = match_keypoints(
        keypoints_l, keypoints_r, descriptors_l, descriptors_r, ratio
    )
    print("Number of matching points:", len(matches_pos))
    print("Step 3 - RANSAC algorithm to find the best homography...")
    homography = ransac(matches_pos)
    print("Step 4 - Create panoramic image...")
    warp_img = warp([img_l, img_r], homography)

    return warp_img


def match_keypoints(keypoints_l, keypoints_r, descriptors_l, descriptors_r, ratio):
    pair_points = []
    for i in trange(len(keypoints_l)):
        dists = np.linalg.norm(descriptors_r - descriptors_l[i], axis=1)
        parted_idx = np.argpartition(dists, 2)[:2]
        if dists[parted_idx[0]] < ratio * dists[parted_idx[1]]:
            pair_points.append([keypoints_l[i].pt, keypoints_r[parted_idx[0]].pt])
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


def warp(imgs, H):
    img_l, img_r = imgs
    hl, wl = img_l.shape[:2]
    hr, wr = img_r.shape[:2]
    stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="int")

    # Compute the inverse of the homography matrix for coordinate mapping
    inv_H = np.linalg.inv(H)

    # Map pixels from the right image to the stitched image using the inverse homography
    for i in trange(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            coor = np.array([j, i, 1])
            img_right_coor = inv_H @ coor
            img_right_coor /= img_right_coor[2]

            x, y = int(round(img_right_coor[1])), int(
                round(img_right_coor[0])
            )  # Map coordinates

            # Skip pixels outside the boundaries of the right image
            if 0 <= x < hr and 0 <= y < wr:
                stitch_img[i, j] = img_r[x, y]

    # Blend the left and transformed right image
    merged = linear_blending([img_l, stitch_img])
    return remove_black_border(merged)


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


def main():
    os.makedirs("output/merged", exist_ok=True)
    os.makedirs("output/keypoints", exist_ok=True)
    files = [
        f
        for f in sorted(os.listdir(FOLDER_PATH))
        if os.path.isfile(os.path.join(FOLDER_PATH, f))
    ]
    print("Files to process:", files)
    file_pairs = [(files[i], files[i + 1]) for i in range(0, len(files), 2)]

    for fname1, fname2 in file_pairs:
        print("=" * 50)
        print(f"Processing {fname1} and {fname2}...")
        output_fname = f"{''.join(fname1.split('.')[:-1])}_{''.join(fname2.split('.')[:-1])}_merge.jpg"

        img_left = cv2.imread(os.path.join(FOLDER_PATH, fname1))
        img_right = cv2.imread(os.path.join(FOLDER_PATH, fname2))

        warp_img = stitch([img_left, img_right], fname1, fname2)

        # Ensure the image is in uint8 format for proper display
        if warp_img.dtype != np.uint8:
            warp_img = np.clip(warp_img, 0, 255).astype(np.uint8)

        merge_img = np.hstack([img_left, img_right, warp_img])
        cv2.imwrite(os.path.join("output", output_fname), warp_img)

        cv2.imwrite(os.path.join("output/merged", output_fname), merge_img)
        print(f"Saved to output/{output_fname}")


if __name__ == "__main__":
    main()
