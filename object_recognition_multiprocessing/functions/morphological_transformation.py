import numpy as np
import cv2


def kernel_size(ratio, img):
    r = round(img.shape[0] / 591 * ratio)
    return r, r


def morph_transformation(img):
    # morphological transformation in order to find regions containing hotpoints
    kernel = np.ones(kernel_size(15, img))
    hotpoins_image_after_elaboration = cv2.dilate(img, kernel)

    kernel = np.ones(kernel_size(10, img))
    hotpoins_image_after_elaboration = cv2.erode(hotpoins_image_after_elaboration, kernel)

    kernel = np.ones(kernel_size(8, img))
    hotpoins_image_after_elaboration = cv2.erode(hotpoins_image_after_elaboration, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size(5, img))
    hotpoins_image_after_elaboration = cv2.dilate(hotpoins_image_after_elaboration, kernel)

    return hotpoins_image_after_elaboration
