import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np


def task1():
    # Set some constants needed in the code
    filenames = ["000001", "000002", "000003"]
    n = len(filenames)

    plt.figure(figsize=(5 * n, 5))
    for i, name in enumerate(filenames):
        img = cv2.imread("img/first task/C0_{0}.png".format(name), cv2.IMREAD_GRAYSCALE)
        # Fruit segmentation
        mask = np.copy(img)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        ret, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask + 1
        cv2.floodFill(mask, None, (0, 0), 0)
        # Edge detection
        edge = cv2.Canny(img, 50, 230)
        edge = edge * mask
        # Show result
        img = cv2.imread("img/first task/C1_{0}.png".format(name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, _ = img.shape
        for j in range(x):
            for k in range(y):
                if edge[j, k] != 0:
                    img[j, k, 0] = 0
                    img[j, k, 1] = 0
                    img[j, k, 2] = 255
        plt.subplot(1, n, i + 1).axis("off")
        plt.imshow(img)
    plt.show()


def task2():
    # Set some constants needed in the code
    filenames = ["000004", "000005"]
    n = len(filenames)

    plt.figure(figsize=(5 * n, 5))
    for i, name in enumerate(filenames):
        # Import images
        img = cv2.imread("img/second task/C0_{0}.png".format(name), cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.imread("img/second task/C1_{0}.png".format(name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Fruit segmentation
        mask = np.copy(img)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        ret, mask = cv2.threshold(mask, 75, 1, cv2.THRESH_BINARY)
        mask = mask + 1
        cv2.floodFill(mask, None, (0, 0), 0)
        x, y = mask.shape
        for j in range(x):
            for k in range(y):
                if mask[j, k] == 2:
                    mask[j, k] = 1
        # Apply mask
        for j in range(img_bgr.shape[2]):
            img_bgr[:, :, j] *= mask
        # Get L channel of LUV color space
        img_l = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Luv)[:, :, 0]
        # Adaptive thresholding with 99x99 neighbourhood
        img_l = cv2.adaptiveThreshold(img_l, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 99, 2)
        # Opening by a cross structuring element to remove little russet areas that are a regular part of apple's
        # texture
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
        img_l = cv2.morphologyEx(img_l, cv2.MORPH_OPEN, kernel)
        # Erosion of the binary mask. It will be used to remove backgrounds point that are still present
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=1)
        # Show result
        x, y = img_l.shape
        for j in range(x):
            for k in range(y):
                if img_l[j, k] == 1 and mask[j, k] == 1:
                    img_rgb[j, k, :] = [255, 0, 0]
        plt.subplot(1, n, i + 1)
        plt.imshow(img_rgb)
    plt.show()


def task3():
    # Set some constants needed in the code
    filenames = ["000006", "000007", "000008", "000009", "000010"]
    n = len(filenames)
    plt.figure(figsize=(5 * n, 5))
    for i, name in enumerate(filenames):
        # Import images
        img = cv2.imread("img/final challenge/C0_{0}.png".format(name), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(
            cv2.imread("img/final challenge/C1_{0}.png".format(name)),
            cv2.COLOR_BGR2RGB)
        # Segmentation
        img = cv2.GaussianBlur(img, (5, 5), 0)
        ret, mask = cv2.threshold(img, 70, 1, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 18))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel)
        cv2.floodFill(mask, None, (0, 0), 1)
        is_bad = np.all(mask)
        x, y = mask.shape
        for j in range(x):
            for k in range(y):
                if mask[j, k] == 0:
                    img_rgb[j, k, 0] = 255
                    img_rgb[j, k, 1] = 0
                    img_rgb[j, k, 2] = 0
        title = "Good Kiwi" if is_bad else "Bad Kiwi"
        plt.subplot(1, n, i + 1).axis("off")
        plt.title(title)
        plt.imshow(img_rgb)
    plt.show()


def main(argv):
    n = len(argv)
    if n < 2:
        print("Error: you need to specify which task you want to run.\n"
              "1, 2 and 3 are possible choices")
        return 1
    task = argv[1]
    if task == "1":
        task1()
    elif task == "2":
        task2()
    elif task == "3":
        task3()
    else:
        print("Error: The task you have specified does not exist.\n"
              "1, 2 and 3 are possible choices")
        return 1
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = sys.argv
    main(args)
