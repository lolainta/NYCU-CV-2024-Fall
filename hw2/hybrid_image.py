import cv2
import numpy as np
import os
from utils import get_image_pairs, add_cutoff_label
from tqdm import trange, tqdm

FOLDER_PATH = "./data/task1and2_hybrid_pyramid/"
FOLDER_PATH = "./my_data/task1and2_hybrid_pyramid/"


def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return 255 * (img - img_min) / (img_max - img_min)


def hybrid_channel(
    img1_channel: np.ndarray,
    img2_channel: np.ndarray,
    cutoff: tuple[int, int] = (20, 20),
    gaussian: bool = True,
) -> np.ndarray:

    # Step 1: Apply Fourier Transform
    rows, cols = img1_channel.shape
    f_img1 = np.fft.fft2(img1_channel)
    f_img2 = np.fft.fft2(img2_channel)

    # Step 2: Shift the zero frequency component to the center
    f_img1 = np.fft.fftshift(f_img1)
    f_img2 = np.fft.fftshift(f_img2)

    # Step 3: Apply filters
    crow, ccol = rows // 2, cols // 2
    dl, dh = cutoff

    def distance(x, y) -> float:
        return (x - crow) ** 2 + (y - ccol) ** 2

    def low_pass_filter(dl: float, gaussian=True) -> np.ndarray:
        if gaussian:
            if dl == 0:
                return np.zeros((rows, cols))
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            return np.exp(-distance(X, Y) / (2 * dl**2))
        else:
            ret = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    # if distance(i, j) <= dl**2:
                    if distance(i, j) <= dl:
                        ret[i, j] = 1
            return ret

    def high_pass_filter(dh: float, gaussian=True) -> np.ndarray:
        if gaussian:
            if dh == 0:
                return np.ones((rows, cols))
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            return 1 - np.exp(-distance(X, Y) / (2 * dh**2))
        else:
            ret = np.ones((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    # if distance(i, j) <= dh**2:
                    if distance(i, j) <= dh:
                        ret[i, j] = 0
            return ret

    f_img1_filtered = f_img1 * low_pass_filter(dl, gaussian)
    f_img2_filtered = f_img2 * high_pass_filter(dh, gaussian)

    # Step 4: Shift the zero frequency component back to the top-left corner
    f_img1_filtered = np.fft.ifftshift(f_img1_filtered)
    f_img2_filtered = np.fft.ifftshift(f_img2_filtered)

    # Step 5: Apply Inverse Fourier Transform
    img1_back = np.fft.ifft2(f_img1_filtered)
    img2_back = np.fft.ifft2(f_img2_filtered)

    # Step 6: Get the real part of the image
    img1_back = np.real(img1_back)
    img2_back = np.real(img2_back)

    # Add the two resulting images to generate the hybrid image
    hybrid_channel = img1_back + img2_back
    hybrid_channel = normalize(hybrid_channel)
    return hybrid_channel.astype(np.uint8)


def hybrid_image(
    img1, img2, cutoff: tuple[int, int] = (20, 20), gaussian: bool = True
) -> np.ndarray:

    assert img1.shape == img2.shape, "Images must have the same size"
    ret = np.zeros_like(img1)
    for channel in range(3):  # 0: B, 1: G, 2: R
        img1_channel = img1[:, :, channel]
        img2_channel = img2[:, :, channel]
        ret[:, :, channel] = hybrid_channel(
            img1_channel,
            img2_channel,
            cutoff=cutoff,
            gaussian=gaussian,
        )
    return ret


def main():
    images = get_image_pairs(FOLDER_PATH)
    os.makedirs("./output/hybrid", exist_ok=True)
    cutoffs: dict[int, tuple[bool, tuple[int, int]]] = {
        0: (False, (15, 15)),
        1: (True, (40, 40)),
        2: (False, (15, 15)),
        3: (True, (30, 30)),
        4: (True, (15, 15)),
        5: (True, (23, 23)),
        6: (True, (55, 55)),
        7: (True, (15, 15)),
        8: (True, (15, 15)),
    }

    for num, (img1, img2) in images:
        print(f"Processing pair {num}")

        cutoff = cutoffs[num][1] if num in cutoffs else (20, 20)
        gaussian = cutoffs[num][0] if num in cutoffs else True
        final = hybrid_image(
            img1,
            img2,
            cutoff=cutoff,
            gaussian=gaussian,
        )
        merged = np.hstack((img1, img2, final))

        # Save the hybrid image
        cv2.imwrite(f"./output/hybrid/{num}.png", final)
        # cv2.imshow("Hybrid Image", merged)
        # cv2.waitKey(0)


def expirements():
    os.makedirs("./output/hybrid", exist_ok=True)
    images = get_image_pairs(FOLDER_PATH)
    for num, (img1, img2) in images:
        results = []
        for gaussian in tqdm([True, False]):
            row = []
            for cut_off_o in trange(2, 10):
                cut_off = cut_off_o**3 // 4
                final = hybrid_image(
                    img1,
                    img2,
                    cutoff=(cut_off, cut_off),
                    gaussian=gaussian,
                )
                final = add_cutoff_label(final, gaussian, cut_off)
                row.append(final)
            results.append(np.hstack(row))
        merged = np.vstack(results)
        cv2.imwrite(f"./output/hybrid/{num}-all.png", merged)


if __name__ == "__main__":
    main()
    # expirements()
