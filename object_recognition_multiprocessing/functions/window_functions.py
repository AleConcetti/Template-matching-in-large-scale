import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt, patches

from objects.constants import Constants


def window_filter(good_matches, test_keypoints, template_keypoints, window):
    dst_pts = []
    src_pts = []
    inliers_good_matches = []

    for good_match in good_matches:
        dst_pt = test_keypoints[good_match.queryIdx].pt
        src_pt = template_keypoints[good_match.trainIdx].pt
        point = Point(dst_pt)

        if point.within(window):
            dst_pts.append(dst_pt)
            src_pts.append(src_pt)
            inliers_good_matches.append(good_match)

    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)

    return inliers_good_matches, dst_pts, src_pts


def pre_filter_second_homography(inliers_good_matches, src_pts, dst_pts, inliers_mask):
    # Create a list representing all the inliers of the retrieved homography
    matches_mask = inliers_mask.ravel().tolist()

    in_dst_pts = []
    in_src_pts = []
    in_inliers_good_matches = []
    for i in range(len(src_pts)):
        if matches_mask[i]:
            in_dst_pts.append(dst_pts[i])
            in_src_pts.append(src_pts[i])
            in_inliers_good_matches.append(inliers_good_matches[i])

    dst_pts = np.float32(in_dst_pts).reshape(-1, 1, 2)
    src_pts = np.float32(in_src_pts).reshape(-1, 1, 2)
    inliers_good_matches = in_inliers_good_matches

    return inliers_good_matches, src_pts, dst_pts


def draw_polygon_on_image(polygon: Polygon, image, scale=1, color='r'):
    minx, miny, maxx, maxy = polygon.bounds
    width = (maxx - minx) * scale
    height = (maxy - miny) * scale

    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Create a Rectangle patch
    rect = patches.Rectangle((minx, miny), width, height, linewidth=1, edgecolor=color, facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

    return


def get_window_from_first_polygon(polygon: Polygon, quadrante, scale=2):
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx)
    dy = (maxy - miny)

    if quadrante == 1:
        # 1-----------2
        # |      | obj|
        # |      |----|
        # |           |
        # |  MASK     |
        # 4-----------3
        p1 = (maxx - scale * dx, miny)
        p2 = (maxx, miny)
        p3 = (maxx, miny + scale * dy)
        p4 = (maxx - scale * dx, miny + scale * dy)
    elif quadrante == 2:
        # 1----------2
        # |obj |      |
        # |----|      |
        # |           |
        # |   MASK    |
        # 4----------3-
        p1 = (minx, miny)
        p2 = (minx + scale * dx, miny)
        p3 = (minx + scale * dx, miny + scale * dy)
        p4 = (minx, miny + scale * dy)
    elif quadrante == 3:
        # 1-----------2
        # |    MASK   |
        # |           |
        # |----|      |
        # |obj |      |
        # 4-----------3
        p1 = (minx, maxy - scale * dy)
        p2 = (minx + scale * dx, maxy - scale * dy)
        p3 = (minx + scale * dx, maxy)
        p4 = (minx, maxy)
    elif quadrante == 4:
        # 1-----------2
        # |   MASK    |
        # |           |
        # |      |----|
        # |      |obj |
        # 4-----------3
        p1 = (maxx - scale * dx, maxy - scale * dy)
        p2 = (maxx, maxy - scale * dy)
        p3 = (maxx, maxy)
        p4 = (maxx - scale * dx, maxy)
    else:
        raise ValueError('Quadrante must be between 1 and 4')

    return Polygon((p1, p2, p3, p4))


def __create_line(x1, y1, x2, y2):
    if x1 == x2 and y1 == y2:
        raise ValueError("Infinite lines")
    elif x1 == x2:
        raise ValueError("Equation type x=q")
    else:
        m = float(y2 - y1) / (x2 - x1)
        q = y1 - (m * x1)

    def line(x):
        return m * x + q

    return line


def __create_non_linear_model(area_tot, lower_bound, upper_bound):
    '''
                         _____up_bound____
                        /
                       /
    __low_bound_______/

    :param area_tot: area of test image
    :param lower_bound:
    :param upper_bound:
    :return: the non linear function drawn above
    '''

    # First point of the line
    x1, y1 = 0.5 * area_tot, lower_bound
    # Second point of the line
    x2, y2 = 0.9 * area_tot, upper_bound

    line = __create_line(x1, y1, x2, y2)

    def non_lin(area):

        value = line(area)

        # Saturation low
        if value < lower_bound:
            value = lower_bound

        # Saturation up
        if value > upper_bound:
            value = upper_bound

        return round(value)

    return non_lin


def get_continually_discarded_constant(area, area_tot):
    non_lin_model = __create_non_linear_model(area_tot, lower_bound=Constants.CONTINUOUSLY_DISCARDED_LOWER_BOUND,
                                              upper_bound=Constants.CONTINUOUSLY_DISCARDED_UPPER_BOUND)

    return non_lin_model(area)


def get_min_match_count_constant(area, area_tot):
    non_lin_model = __create_non_linear_model(area_tot, lower_bound=Constants.MIN_MATCH_COUNT_LOWERBOUND,
                                              upper_bound=Constants.MIN_MATCH_COUNT_UPPERBOUND)

    return non_lin_model(area)


def get_max_iters_constant(cont_discard_count, max_cont_discard, number_good_matches, min_match_count):
    delta = Constants.MAX_MAX_ITERS - Constants.MIN_MAX_ITERS
    penalty_interval = delta / max_cont_discard
    # number_good_matches = (min_match_count, +inf)
    # Hence, 1/number_good_matches = (0, 1 / min_match_count)
    # And so, min_match_count/number_good_matches = (0, 1)
    # when this ratio is near one is probably that cv2.findhomography is not going to find nothing because
    # number_of_good_matches is very close to min_match_count
    # So I can increment the cont_discatd_count by maximum 10% of the max cont_discard
    count_increment = (0.1 * max_cont_discard) * min_match_count / number_good_matches
    total_count = cont_discard_count + count_increment

    value2return = np.round(Constants.MAX_MAX_ITERS - total_count * penalty_interval).astype(int)

    if value2return > Constants.MAX_MAX_ITERS:
        return Constants.MAX_MAX_ITERS
    elif value2return < Constants.MIN_MAX_ITERS:
        return Constants.MIN_MAX_ITERS
    else:
        return value2return
