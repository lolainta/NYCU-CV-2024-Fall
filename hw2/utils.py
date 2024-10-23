import os
import re
import cv2
import numpy as np


def get_files(folder_path: str) -> dict[int, list[str]]:
    files = os.listdir(folder_path)
    pattern = re.compile(r"^(\d+)")
    image_pairs = {}
    for file in files:
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            if num not in image_pairs:
                image_pairs[num] = []
            image_pairs[num].append(file)
    return image_pairs


def get_image_pairs(folder_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    image_pairs: dict[int, list[str]] = get_files(folder_path)
    ret = []
    for num, pair in image_pairs.items():
        if len(pair) != 2:
            print(f"Pair {num} has {len(pair)} images: {pair}")

        img1_path = pair[0]
        img2_path = pair[1]
        print(f'Reading image pair {num}: ("{img1_path}", "{img2_path}")')

        # Read the images
        img1 = cv2.imread(os.path.join(folder_path, img1_path))
        img2 = cv2.imread(os.path.join(folder_path, img2_path))

        # Check if the two images have the same size
        if img1.shape != img2.shape:
            print(f'\tResizing "{img2_path}" to match "{img1_path}"')
            print(f"\tOriginal size: {img2.shape}, Target size: {img1.shape}")
            img2 = cv2.resize(
                img2, (img1.shape[1], img1.shape[0])
            )  # Resize img2 to match img1

        ret.append((num, (img1, img2)))
    return ret


def get_images(folder_path: str) -> list[str]:
    files = os.listdir(folder_path)
    images = []
    for file in files:
        if file.lower().endswith((".jpg", ".bmp")):
            images.append(
                (
                    file,
                    cv2.imread(
                        os.path.join(folder_path, file),
                        cv2.IMREAD_GRAYSCALE,
                    ),
                )
            )
    return images


def split_image(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    print(f"Reading {file_name} with shape {img.shape}")
    height = img.shape[0] // 3

    # Separate the color channels
    b = img[:height]
    g = img[height : 2 * height]
    r = img[2 * height : 3 * height]
    return b, g, r


def get_prokudin_gorsky_images(
    folder_path: str,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    files = os.listdir(folder_path)
    images = []
    for file in files:
        if file.lower().endswith((".jpg", ".bmp", ".png", ".tif")):
            b, g, r = split_image(os.path.join(folder_path, file))
            images.append((file, b, g, r))
    return images


def add_gt_label(gt, gt_offset):
    cv2.putText(
        gt,
        f"Ground Truth",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        gt,
        f"Blue-Green Offset: {gt_offset[0][0]} {gt_offset[0][1]}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        gt,
        f"Red-Green Offset: {gt_offset[1][0]} {gt_offset[1][1]}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return gt


def add_results_label(result, bo_shift, ro_shift, NCC):
    cv2.putText(
        result,
        f"NCC: {NCC:.4f}",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"Blue-Green Offset: {bo_shift[1]} {bo_shift[0]}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"Red-Green Offset: {ro_shift[1]} {ro_shift[0]}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return result
