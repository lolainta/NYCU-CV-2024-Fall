import cv2
import numpy as np
import os
from tqdm import trange, tqdm
import time

from icecream import ic

from camera_calibration_show_extrinsics import plot_extrinsics

np.set_printoptions(suppress=True, linewidth=200, threshold=10)
ic.configureOutput(includeContext=True)
# ic.disable()

CASE = "room"
# CASE = "park"

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
META_DIR = os.path.join(DATA_DIR, CASE)


OUTPUT_DIR = os.path.join("output", CASE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def warp(i, imgs, H):
    img_l, img_r = imgs
    # ic(img_l, img_r, H)
    hl, wl = img_l.shape[:2]
    hr, wr = img_r.shape[:2]
    # inv_H = np.linalg.inv(H)

    stitch_img = np.zeros((hl + hr, wl + wr, 3), dtype="int")

    height_s, width_s = stitch_img.shape[:2]
    height_r, width_r = img_r.shape[:2]

    # 1. Create a grid of (i, j) for the entire stitched image
    i_grid, j_grid = np.indices(
        (height_s, width_s)
    )  # i_grid.shape == j_grid.shape == (height_s, width_s)
    ic(i_grid.shape, j_grid.shape)

    # 2. Convert (i, j) -> homogeneous form: (j, i, 1)
    #    Note: j is the x-coordinate, i is the y-coordinate in image space.
    coor_grid = np.stack([j_grid, i_grid, np.ones_like(i_grid)], axis=-1)
    ic(coor_grid.shape)
    # coor_grid.shape = (height_s, width_s, 3)

    # 3. Apply the inverse homography to every pixel in one shot
    #    coor_grid @ inv_H.T => shape: (height_s, width_s, 3)
    img_right_coor = coor_grid @ H.T

    # 4. Normalize by the 3rd component (homogeneous coordinate)
    img_right_coor /= img_right_coor[..., 2][
        ..., None
    ]  # broadcast across last dimension
    ic(img_right_coor.shape)

    # 5. Extract (x, y) in integer pixel space
    #    Note that we do y = round(...) of the 0th coordinate,
    #    and x = round(...) of the 1st coordinate if you follow (row, col) == (x, y).
    x = np.rint(img_right_coor[..., 1]).astype(int)  # row index
    y = np.rint(img_right_coor[..., 0]).astype(int)  # column index
    ic(x.shape, y.shape)

    # 6. Build a mask for valid (x, y) within the right image boundaries
    valid_mask = (x >= 0) & (x < height_r) & (y >= 0) & (y < width_r)
    ic(valid_mask.shape)

    # 7. Assign the valid pixels from img_r into stitch_img
    #    We use fancy indexing on both sides.
    stitch_img[i_grid[valid_mask], j_grid[valid_mask]] = img_r[
        x[valid_mask], y[valid_mask]
    ]

    cv2.imwrite(f"test/stitch_img{i}.jpg", stitch_img)

    # Lastly, perform your blending step
    merged = linear_blending([img_l, stitch_img])
    return remove_black_border(merged)


def warp_t(imgs, H):
    img_l, img_r = imgs
    ic(img_l, img_r, H)
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
            x, y = np.rint(img_right_coor[1]), np.rint(img_right_coor[0])

            # Skip pixels outside the boundaries of the right image
            if 0 <= x and x < hr and 0 <= y < wr:
                ic(x, y, i, j)
                stitch_img[i, j] = img_r[x, y]
    ic(stitch_img.shape)
    # Blend the left and transformed right image
    merged = linear_blending([img_l, stitch_img])

    return merged
    ic(merged.shape)
    # return remove_black_border(merged)


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


def compute_homography(intrinsic, ex1, ex2):
    # ex1, ex2: 3x4 extrinsic matrices
    K = np.array(intrinsic).reshape(3, 3)
    K_inv = np.linalg.inv(K)
    ex1_inv = np.linalg.pinv(ex1)
    ex2_inv = np.linalg.pinv(ex2)
    H = K @ ex2 @ ex1_inv @ K_inv
    return H / H[2, 2]
    # Intrinsic matrix
    K = np.array(intrinsic).reshape(3, 3)

    # Extract rotation and translation from extrinsics
    R1, t1 = np.array(ex1[:, :3]).reshape(3, 3), np.array(ex1[:, 3:]).reshape(3)
    R2, t2 = np.array(ex2[:, :3]).reshape(3, 3), np.array(ex2[:, 3:]).reshape(3)
    ic(R1, t1)
    ic(R2, t2)

    # Relative rotation and translation
    R_rel = R2 @ np.linalg.inv(R1)
    t_rel = t2 - (R_rel @ t1)

    # Plane parameters (assuming z=0 for a planar homography)
    n = np.array([0, 0, 1])  # Plane normal
    d = 1.0  # Distance of plane from origin

    # Compute homography
    H = K @ (R_rel - (t_rel[:, None] * n / d)) @ np.linalg.inv(K)
    return H


def test1(images, intrinsic, extrinsics):
    out_dir = "test"
    os.makedirs(out_dir, exist_ok=True)

    pano = np.zeros((5000, 8000, 3), dtype=np.uint8)

    for i, ((img_name, image), extrinsic) in enumerate(zip(images, extrinsics)):
        ic(img_name)
        if i == 0:
            pano[0 : image.shape[0], 0 : image.shape[1]] = image
            continue
        ex1 = extrinsics[0]  # 3x4
        ex2 = extrinsic  # 3x4
        H = compute_homography(intrinsic, ex1, ex2)
        ic(H)
        warp_img = warp(i, [images[0][1], image], H)
        ic(warp_img.shape)

        pano = linear_blending([pano, warp_img])

        cv2.imwrite(f"{out_dir}/warp{i}.jpg", warp_img)
        cv2.imwrite(f"{out_dir}/pano{i}.jpg", pano)
        ic(f"{out_dir}/{i}.jpg")

    cv2.imwrite(f"{out_dir}/panorama.jpg", pano)

def test2(images, checkboard_corners):

    PANO_WIDTH = 8000
    PANO_HEIGHT = 4000
    MIN_WIDTH = PANO_WIDTH // 2 - 100
    MAX_WIDTH = PANO_WIDTH // 2 + 100
    MIN_HEIGHT = PANO_HEIGHT // 2 - 100
    MAX_HEIGHT = PANO_HEIGHT // 2 + 100


    fixed_cords = [
        [MIN_WIDTH, MIN_HEIGHT],
        [MAX_WIDTH, MIN_HEIGHT],
        [MIN_WIDTH, MAX_HEIGHT],
        [MAX_WIDTH, MAX_HEIGHT],
    ]

    if CASE == "room":
        fixed_cords = fixed_cords[0], fixed_cords[2], fixed_cords[1], fixed_cords[3]
    elif CASE == "park":
        fixed_cords = fixed_cords[3], fixed_cords[2], fixed_cords[1], fixed_cords[0]
    else:
        raise ValueError("Invalid CASE")
    pano = np.zeros((PANO_HEIGHT, PANO_WIDTH, 3, len(images)), dtype=np.uint8)

    def align(img_name, image, corner, fixed_cords):
        # ic(img_name)
        M = cv2.getPerspectiveTransform(
            np.array(corner, dtype=np.float32),
            np.array(fixed_cords, dtype=np.float32),
        )
        # ic(M)
        ret = cv2.warpPerspective(image, M, (PANO_WIDTH, PANO_HEIGHT))
        # ic(ret.shape)
        cv2.imwrite(f"{OUTPUT_DIR}/{img_name}", ret)
        return ret


    for i, ((img_name, image), corner) in enumerate(zip(images, checkboard_corners)):
        ic(img_name)
        align(img_name, image, corner, fixed_cords)
    for i, (img_name, _) in tqdm([*enumerate(images)]):
        img = cv2.imread(f"{OUTPUT_DIR}/{img_name}")
        pano[:, :, :, i] = img
    
    print("Merging images...")
    panorama = np.true_divide(pano.sum(axis=-1), (pano != 0).sum(axis=-1))
    cv2.imwrite(f"{OUTPUT_DIR}/panorama.jpg", panorama)
    print(f"Panorama saved to {OUTPUT_DIR}/panorama.jpg")

    # for i, ((img_name, image), corner) in enumerate(zip(images, checkboard_corners)):
    #     warp_img = align(img_name, image, corner, fixed_cords)
    #     cv2.imwrite(f"{OUTPUT_DIR}/warp{i}.jpg", warp_img)
    #     pano[:, :, :, i] = warp_img
    #     ic(f"{OUTPUT_DIR}/{i}.jpg")

    # pano = np.true_divide(pano.sum(axis=-1), (pano != 0).sum(axis=-1))
    # cv2.imwrite(f"{OUTPUT_DIR}/panorama.jpg", pano)


def main():
    # INTRISIC_PATH = os.path.join(META_DIR, "intrinsic.txt")
    # intrinsic = np.array(
    #     [line.split() for line in open(INTRISIC_PATH).readlines()], dtype=np.float32
    # )
    # ic(intrinsic)

    # EXTRINSIC_PATH = os.path.join(META_DIR, "extrinsics.txt")
    # extrinsic_pair = [
    #     (line.split()[0], np.array(line.split()[1:], dtype=np.float32))
    #     for line in open(EXTRINSIC_PATH).readlines()
    # ]
    # images = [
    #     (fname, cv2.imread(os.path.join(DATA_PATH, fname), cv2.IMREAD_COLOR))
    #     for fname, _ in extrinsic_pair
    # ]
    # extrinsics = np.array(
    #     [
    #         np.hstack([cv2.Rodrigues(ex[:3])[0], ex[3:].reshape(3, 1)])
    #         for _, ex in extrinsic_pair
    #     ]
    # )
    # ic(extrinsics)

    # plot_extrinsics(intrinsic, extrinsics)
    # warp_img = warp(images, intrinsic, extrinsics)
    # test1(images, intrinsic, extrinsics)


    IMGPOINTS_PATH = os.path.join(META_DIR, "imgpoints.txt")
    checkboard_pair = [
        (line.split()[0], np.array(line.split()[1:], dtype=np.float32))
        for line in open(IMGPOINTS_PATH).readlines()
    ]
    images = [
        (fname, cv2.imread(os.path.join(IMAGE_DIR, fname), cv2.IMREAD_COLOR))
        for fname, _ in checkboard_pair
    ]

    checkboard_corners = np.array(
        [np.array(pts).reshape(-1, 2) for _, pts in checkboard_pair]
    )
    ic(checkboard_corners)
    time.sleep(1)
    test2(images, checkboard_corners)


if __name__ == "__main__":
    main()
