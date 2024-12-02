import numpy as np


def load(case: int):
    match case:
        case 1:
            IMG_PATH = ["data/Mesona1.JPG", "data/Mesona2.JPG"]
            K = np.array(
                [
                    [
                        [1.4219, 0.0005, 0.5092],
                        [0, 1.4219, 0],
                        [0, 0, 0.0010],
                    ],
                    [
                        [1.4219, 0.0005, 0.5092],
                        [0, 1.4219, 0],
                        [0, 0, 0.0010],
                    ],
                ]
            )
            OUTPUT_PATH = "output/mesona"

        case 2:
            IMG_PATH = ["data/Statue1.bmp", "data/Statue2.bmp"]
            K = np.array(
                [
                    [
                        [5426.566895, 0.678017, 330.096680],
                        [0, 5423.133301, 648.950012],
                        [0, 0, 1],
                    ],
                    [
                        [5426.566895, 0.678017, 387.430023],
                        [0, 5423.133301, 620.616699],
                        [0, 0, 1],
                    ],
                ]
            )
            OUTPUT_PATH = "output/statue"
        case _:
            raise ValueError(f"Invalid case: {case}")
    return IMG_PATH, K, OUTPUT_PATH