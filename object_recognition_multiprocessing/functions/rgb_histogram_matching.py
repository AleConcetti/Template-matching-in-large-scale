# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Perform histogram matching over each images channel
from objects.homography import Homography


def rgb_histogram_matching(source, template):
    # Extract channels from images
    sb, sg, sr = cv2.split(source)
    tb, tg, tr = cv2.split(template)
    
    # Perform histogram matching over each channel
    matched_blue = __hist_match(sb, tb)
    matched_green = __hist_match(sg, tg)
    matched_red = __hist_match(sr, tr)
    
    return cv2.merge((matched_blue, matched_green, matched_red))
    

# Perform histogram matching between two grayscale images
def __hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape).astype(np.uint8)


def evaluete_homography(homography: Homography, test_image, plot=False, rgb_adj=False):
    template_image = homography.template.image

    # Invert the homography transformation
    H_inv = np.linalg.inv(homography.H)

    # Rectify the test image and crop the interesting part
    height, width = template_image.shape[0:2]
    rect_test_image = cv2.warpPerspective(test_image, H_inv, (width, height))

    # Match the histograms
    if rgb_adj:
        matched_rect_test_image = rect_test_image
    else:
        matched_rect_test_image = rgb_histogram_matching(rect_test_image, template_image)

    # Compute the difference between template and histogram matched rectified test image
    abs_diff_image = cv2.absdiff(template_image,
                                 matched_rect_test_image)

    # Compute the image representing the pixelwise difference norm
    abs_diff_norm_image = np.linalg.norm(abs_diff_image, axis=2).astype(np.uint8)

    error_mean = abs_diff_norm_image.mean()
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.suptitle('Mean error: {}'.format(error_mean))
        ax1.imshow(template_image)
        ax2.imshow(matched_rect_test_image)
        ax3.imshow(abs_diff_image)
        ax4.imshow(abs_diff_norm_image)
        plt.show()

    return error_mean, template_image, matched_rect_test_image, abs_diff_norm_image

def best_homography(h1: Homography, h2: Homography, test_image, plot=False):

    error1, template_image1, matched_rect_test_image1, abs_diff_norm_image1 = evaluete_homography(h1, test_image)
    error2, template_image2, matched_rect_test_image2, abs_diff_norm_image2 = evaluete_homography(h2, test_image)
    if plot:
        fig, axis = plt.subplots(2, 3)
        fig.suptitle('Error1: {}. Error2: {}. '.format(error1, error2))
        axis[0][0].imshow(template_image1)
        axis[0][1].imshow(matched_rect_test_image1)
        axis[0][2].imshow(abs_diff_norm_image1)

        axis[1][0].imshow(template_image2)
        axis[1][1].imshow(matched_rect_test_image2)
        axis[1][2].imshow(abs_diff_norm_image2)

        plt.show()

    if error1 < error2:
        return h1
    else:
        return h2