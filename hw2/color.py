import numpy as np
import cv2
import os
from utils import get_prokudin_gorsky_images
from tqdm import trange, tqdm


def shift_image(img, off_x, off_y):
    img = np.roll(img, (off_x, off_y), (1, 0))
    if off_x < 0:
        img[:, off_x:] = 128
    else:
        img[:, :off_x] = 128
    if off_y < 0:
        img[off_y:, :] = 128
    else:
        img[:off_y, :] = 128
    return img


def SSD(img1, img2):
    return np.sum((img1 - img2) ** 2)


def NCC(img1, img2):
    img1 = img1 - np.mean(img1, axis=0)
    img2 = img2 - np.mean(img2, axis=0)
    return -np.sum(img1 * img2) / np.sqrt(np.sum(img1**2) * np.sum(img2**2))


def MAE(img1, img2):
    return np.sum(np.abs(img1 - img2))


def calculate_error(img1, img2, off_x, off_y, loss_func):
    img2 = shift_image(img2, off_x, off_y)
    return loss_func(img1, img2)


def crop_image(img, crop_proportion):
    n, m = img.shape
    return img[
        int(crop_proportion * m) : -int(crop_proportion * m),
        int(crop_proportion * n) : -int(crop_proportion * n),
    ]


def align(
    img1, img2, offset_x: tuple[int, int], offset_y: tuple[int, int], loss_func=NCC
) -> tuple[int, int]:
    # Crop the images
    img1_crop = crop_image(img1, 0.1)
    img2_crop = crop_image(img2, 0.1)
    min_error = float("inf")
    best_shift = (None, None)

    # Loop over all the different displacement permutations
    for i in trange(*offset_x):
        for j in range(*offset_y):
            error = calculate_error(img1_crop, img2_crop, i, j, loss_func)
            if error < min_error:
                min_error = error
                best_shift = (i, j)
    assert best_shift[0] != None and best_shift[1] != None
    return best_shift


def align_img(b, g, r, X_OFFSET, Y_OFFSET):
    # Choose the loss function
    # loss_func = SSD
    # loss_func = MAE
    loss_func = NCC
    print(f"Using {loss_func.__name__} as the loss function")

    print("Aligning the blue and green color channels")
    b_shift = align(g, b, X_OFFSET, Y_OFFSET, loss_func)
    print("Aligning the red and green color channels")
    r_shift = align(g, r, X_OFFSET, Y_OFFSET, loss_func)
    return b_shift, r_shift


def main():
    # folder_path = "data/task3_colorizing"
    folder_path = "data/task3_cmu"
    os.makedirs("./results/color", exist_ok=True)
    print("Starting the colorization process")
    for fname, bo, go, ro in get_prokudin_gorsky_images(folder_path):
        print("--------------------")

        b, g, r = bo, go, ro  # save the original images before resizing

        # Resize the images to speed up the process
        resize_ratio = 1
        while b.shape[0] > 512:
            b = cv2.resize(b, (b.shape[1] // 2, b.shape[0] // 2))
            g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
            r = cv2.resize(r, (r.shape[1] // 2, r.shape[0] // 2))
            resize_ratio *= 2
        if resize_ratio != 1:
            print(f"Resized the images by {resize_ratio} times")

        print(f"Processing {fname} with shape {b.shape}")

        assert b.shape == g.shape == r.shape
        n, m = b.shape

        # Align the green and red color channels to the blue color channel using pyramid
        X_OFFSET = (int(-n * 5 / 100), int(n * 5 / 100))
        Y_OFFSET = (int(-m * 5 / 100), int(m * 5 / 100))
        print(
            f"Finding the best alignment for {fname} in the range {X_OFFSET=} {Y_OFFSET=}"
        )

        b_shift, r_shift = align_img(b, g, r, X_OFFSET, Y_OFFSET)

        # Rescale the shifts to the original image size
        bo_shift = (b_shift[0] * resize_ratio, b_shift[1] * resize_ratio)
        ro_shift = (r_shift[0] * resize_ratio, r_shift[1] * resize_ratio)
        print(f"Blue-Green Offset: {bo_shift[1]} {bo_shift[0]}")
        print(f"Red-Green Offset: {ro_shift[1]} {ro_shift[0]}")

        # Shift the images
        abo = shift_image(bo, *bo_shift)
        aro = shift_image(ro, *ro_shift)

        # Stack the color channels
        im_out = np.dstack([abo, go, aro])

        # Convert to uint8 and scale to [0, 255] before saving
        im_out_uint8 = np.clip(im_out, 0, 255).astype(np.uint8)

        # Save the results
        print(f"Saving the results to results/color/{fname[:-4]}-out.png")

        with open(f"./results/color/{fname[:-4]}-out.txt", "w") as f:
            f.write(f"Blue-Green Offset: {bo_shift[1]} {bo_shift[0]}\n")
            f.write(f"Red-Green Offset: {ro_shift[1]} {ro_shift[0]}\n")

        cv2.imwrite(
            f"./results/color/{fname[:-4]}-out.png",
            im_out_uint8,
        )
        # cv2.imshow("image", im_out_uint8)
        # cv2.waitKey(0)


if __name__ == "__main__":
    main()
