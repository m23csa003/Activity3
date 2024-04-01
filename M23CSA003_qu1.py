# %%
#!pip install gif2numpy

# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import gif2numpy


# %%
class Custom_CV:
    @staticmethod
    def padding(img, kernelsize, mode="reflect"):
        # padding
        padx = kernelsize[0] - 1
        pady = kernelsize[1] - 1
        padded_img = np.zeros((img.shape[0] + padx, img.shape[1] + pady))
        padded_img[
            padx // 2 : img.shape[0] + padx // 2, pady // 2 : img.shape[1] + pady // 2
        ] = img

        if mode == "reflect":
            if pady > 0:
                padded_img[:, 0 : pady // 2] = padded_img[:, slice(pady // 2, 0, -1)]
                padded_img[:, -pady // 2 :] = padded_img[
                    :, slice(-pady // 2 - 1, -1, 1)
                ]

            if padx > 0:
                padded_img[0 : padx // 2, :] = padded_img[slice(padx // 2, 0, -1), :]
                padded_img[-padx // 2 :, :] = padded_img[
                    slice(-padx // 2 - 1, -1, 1), :
                ]
        elif mode == "zero":
            pass
        return padded_img

    @staticmethod
    def apply_convolution(img, kernel):
        kernelsize_x, kernelsize_y = kernel.shape
        image_height, image_width = img.shape
        padded_img = Custom_CV.padding(img, kernel.shape)
        conv_img = np.zeros_like(img)

        # convolution
        for i in range(kernelsize_x):
            for j in range(kernelsize_y):
                conv_img += (
                    padded_img[i : image_height + i, j : image_width + j] * kernel[i, j]
                )
        return conv_img

    @staticmethod
    def sobel(img, dir):
        if dir == "x":
            # mn complexity
            # kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            # seperate kernels => m+n complexity
            kernelx = np.array([[-1, 0, 1]])
            kernely = np.array([[1], [2], [1]])
        else:
            # kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
            kernelx = np.array([[1, 2, 1]])
            kernely = np.array([[1], [0], [-1]])

        convx = Custom_CV.apply_convolution(img, kernelx)
        convxy = Custom_CV.apply_convolution(convx, kernely)
        return convxy

    @staticmethod
    def gaussian_blur(img, ksize, sigma):
        if sigma == 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        kernelx = np.zeros((ksize,))
        for i in range(ksize):
            kernelx[i] = np.exp(-((i - 1) ** 2) / (2 * sigma**2))
        kernelx = kernelx.reshape(1, -1)
        kernely = kernelx.reshape(-1, 1)
        convx = Custom_CV.apply_convolution(img, kernelx)
        convxy = Custom_CV.apply_convolution(convx, kernely)
        return convxy

    def non_max_suppression(dst, kernelsize):
        padded_dst = Custom_CV.padding(dst, kernelsize)
        for i in range(dst.shape[0]):
            for j in range(dst.shape[1]):
                if dst[i, j] != np.max(
                    padded_dst[i : i + kernelsize[0], j : j + kernelsize[1]]
                ):
                    dst[i, j] = 0
        return dst

    @staticmethod
    def harris_corners(gray, sobelxx, sobelyy, sobelxy, block_size, k=0.04):
        dist = np.zeros_like(gray, dtype=float)

        pad = block_size // 2
        padded_sobelxx = Custom_CV.padding(
            sobelxx, (block_size, block_size), mode="zero"
        )
        padded_sobelyy = Custom_CV.padding(
            sobelyy, (block_size, block_size), mode="zero"
        )
        padded_sobelxy = Custom_CV.padding(
            sobelxy, (block_size, block_size), mode="zero"
        )
        slice_ = lambda x: slice(x - pad, x + pad + 1)

        for xpixel in tqdm(range(pad, gray.shape[0] + pad)):
            for ypixel in range(pad, gray.shape[1] + pad):
                slice_x = slice_(xpixel)
                slice_y = slice_(ypixel)
                M = np.array(
                    [
                        [
                            np.sum(padded_sobelxx[slice_x, slice_y]),
                            np.sum(padded_sobelxy[slice_x, slice_y]),
                        ],
                        [
                            np.sum(padded_sobelxy[slice_x, slice_y]),
                            np.sum(padded_sobelyy[slice_x, slice_y]),
                        ],
                    ]
                )

                det_M = np.linalg.det(M)
                trace_M = np.trace(M)
                R = det_M - k * (trace_M**2)
                dist[xpixel - pad, ypixel - pad] = R

        dist = Custom_CV.non_max_suppression(dist, (3, 3))
        return dist

    def HarrisCornerDetector(self, img_dir, block_size, k=0.04, ksize=3, imshow=False):
        img = cv.imread(img_dir)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        sobelx = Custom_CV.sobel(gray, "x")
        sobely = Custom_CV.sobel(gray, "y")
        sobelxy = sobelx * sobely
        sobelxx = sobelx**2
        sobelyy = sobely**2
        sobelxx = Custom_CV.gaussian_blur(sobelxx, ksize, 0)
        sobelyy = Custom_CV.gaussian_blur(sobelyy, ksize, 0)
        sobelxy = Custom_CV.gaussian_blur(sobelxy, ksize, 0)

        dst = Custom_CV.harris_corners(gray, sobelxx, sobelyy, sobelxy, block_size, k)
        threshold = np.where(dst > 0.01 * dst.max())
        [
            cv.circle(img, (i, j), 2, (0, 0, 255), -1)
            for j, i in zip(threshold[0], threshold[1])
        ]

        if imshow:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            plt.imshow(img)
        return img, dst


# %%


def openCV_HarrisCornerDetector(image_dir):

    img = cv.imread(image_dir)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img, dst


def Compare_Harris_opencv(imagefolder_dir):
    images_dir = os.listdir(imagefolder_dir)
    for file in images_dir:
        fname = os.path.join(imagefolder_dir, file)
        if fname.endswith(".gif"):
            continue
        img = cv.imread(fname)
        image_opencv, dst_opencv = openCV_HarrisCornerDetector(
            os.path.join(imagefolder_dir, file)
        )
        image_custom, dst_custom = Custom_CV().HarrisCornerDetector(
            os.path.join(imagefolder_dir, file), 2, 0.04, 3
        )
        image_opencv = cv.cvtColor(image_opencv, cv.COLOR_BGR2RGB)
        image_custom = cv.cvtColor(image_custom, cv.COLOR_BGR2RGB)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 3, figsize=(15, 15))
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        ax[1].imshow(image_opencv)
        ax[1].set_title("OpenCV Harris Corner")
        ax[2].imshow(image_custom)
        ax[2].set_title("Custom Harris Corner")
        plt.show()
        # break


# %%
Compare_Harris_opencv(
    r"C:\Users\KHADGA JYOTH ALLI\Desktop\programming\Class Work\IITJ\Computer Vision\Assignment1\Assignment 1 Input Images\Question 1"
)
