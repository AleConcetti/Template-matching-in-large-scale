import cv2
import numpy as np
from shapely.geometry import Polygon

from functions.homography_validation import all_inliers_in_the_polygon, \
    out_area_ratio
from functions.plot_manager import save_two_polygons, save_window, save_inliers
from functions.window_functions import window_filter, \
    get_continually_discarded_constant, get_min_match_count_constant, \
    get_max_iters_constant
from objects.constants import Constants
from objects.homography import Homography


def find_homography_double_check(test_image, template, good_matches, window,
                                 test_keypoints, template_keypoints, plots,
                                 id_hotpoint, id_pos, id_homography, ratios_list, id_hom_global, big_window=False):

    H_found = None
    inliers_found = None
    object_polygon_found = None

    H1, inliers_matches1, object_polygon1, plots = __find_homography(test_image, template, window, good_matches,
                                                                     test_keypoints, template_keypoints, ratios_list,
                                                                     id_homography, big_window, id_hom_global, plots)

    if H1 is None:
        # print('No homography found\n')
        return None, good_matches, ratios_list, plots
    else:
        H_found = H1
        inliers_found = inliers_matches1
        object_polygon_found = object_polygon1

    H2, inliers_matches2, object_polygon2, plots = __find_homography(test_image, template, object_polygon1,
                                                                     good_matches,
                                                                     test_keypoints, template_keypoints, ratios_list,
                                                                     id_homography, big_window, id_hom_global, plots)

    if H2 is not None:
        # print('The second homography not found\n')

        H_found = H2
        inliers_found = inliers_matches2
        object_polygon_found = object_polygon2


    # ------------------------------------------
    #               PLOTS
    # ------------------------------------------
    if Constants.SAVE:
        save_two_polygons(test_image, id_hotpoint, id_pos, id_homography, object_polygon1, object_polygon2,
                          template.name)
        save_window(test_image, id_hotpoint, id_pos, id_homography, window, inliers_found, test_keypoints,
                    template.name)
        save_inliers(test_image, id_hotpoint, id_pos, id_homography, object_polygon_found, inliers_found,
                     test_keypoints, template.name)


    # ------------------------------------------
    #               CLEAN MATCHES
    # ------------------------------------------
    for match in inliers_found:
        good_matches.remove(match)

    # Update ratio
    homography_found = Homography(H_found, object_polygon_found, id_hotpoint, id_pos, id_homography, id_hom_global,
                                  template, len(inliers_found))
    ratios_list.add_new_ratio(homography_found, template)

    return homography_found, good_matches, ratios_list, plots


def __find_homography(test_image, template, window, good_matches, test_keypoints, template_keypoints, ratios_list, idd,
                      big_window, id_hom_global, plots):

    inliers_good_matches, dst_pts, src_pts = window_filter(good_matches, test_keypoints, template_keypoints, window)

    height, width, _ = test_image.shape  # 1000, 750, 3
    test_image_polygon = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    area_test_image = test_image_polygon.area
    area_window = window.area

    continuously_discarded_count = 0

    end = False
    while not end:
        max_continuously_discarded = get_continually_discarded_constant(area_window, area_test_image)
        if continuously_discarded_count < max_continuously_discarded:
            min_match_count = get_min_match_count_constant(area_window, area_test_image)
            if len(good_matches) >= min_match_count:
                if len(src_pts) >= Constants.MIN_MATCH_CURRENT:

                    max_iters = get_max_iters_constant(continuously_discarded_count, max_continuously_discarded,
                                                       len(good_matches), min_match_count)

                    H, inliers_mask = cv2.findHomography(src_pts,
                                                         dst_pts,
                                                         cv2.RANSAC,
                                                         Constants.RANSAC_REPROJECTION_ERROR,
                                                         maxIters=max_iters)

                    # If no available homography exists, the algorithm ends
                    if H is not None:
                        matches_mask = inliers_mask.ravel().tolist()
                        inliers_matches = [match for i, match in enumerate(inliers_good_matches) if matches_mask[i]]

                        height, width = template.image.shape[0:2]

                        src_vrtx = np.float32([[0, 0],
                                               [0, height - 1],
                                               [width - 1, height - 1],
                                               [width - 1, 0]]).reshape(-1, 1, 2)
                        # Extract test image rectangle vertices
                        test_height, test_width = test_image.shape[0:2]
                        test_vrtx = np.float32([[0, 0],
                                                [0, test_height - 1],
                                                [test_width - 1, test_height - 1],
                                                [test_width - 1, 0]])
                        test_image_polygon = Polygon([(test_vrtx[0][0], test_vrtx[0][1]),
                                                      (test_vrtx[1][0], test_vrtx[1][1]),
                                                      (test_vrtx[2][0], test_vrtx[2][1]),
                                                      (test_vrtx[3][0], test_vrtx[3][1])])

                        dst_vrtx = cv2.perspectiveTransform(src_vrtx, H)
                        object_polygon = Polygon([(dst_vrtx[0][0][0], dst_vrtx[0][0][1]),
                                                  (dst_vrtx[1][0][0], dst_vrtx[1][0][1]),
                                                  (dst_vrtx[2][0][0], dst_vrtx[2][0][1]),
                                                  (dst_vrtx[3][0][0], dst_vrtx[3][0][1])])

                        if (np.linalg.matrix_rank(H) == H.shape[0] and
                                object_polygon.is_valid and
                                all_inliers_in_the_polygon(test_keypoints, inliers_matches, object_polygon) and
                                out_area_ratio(object_polygon, test_image_polygon, Constants.OUT_OF_IMAGE_THRESHOLD)):

                            if np.count_nonzero(matches_mask) >= Constants.MIN_MATCH_CURRENT:

                                is_likely, plot = ratios_list.is_homography_likely(object_polygon, template,
                                                                                    test_image, idd,
                                                                                    big_window, id_hom_global)
                                if plot is not None:
                                    plots.append(plot)

                                if is_likely:
                                    return H, inliers_matches, object_polygon, plots
                                else:
                                    # print('Homography discarded because of the ratio of sides')
                                    continuously_discarded_count += 1
                                id_hom_global[0] += 1
                            else:
                                # print("Not enough matches are found in the last homography: {}/{}".format(
                                #     np.count_nonzero(matches_mask), MIN_MATCH_CURRENT))
                                end = True
                        else:
                            # print("Degenerate homography found")
                            continuously_discarded_count += 1
                    else:
                        # print("Not possible to find another homography")
                        end = True
                else:
                    # print("Too few points to fit the homography ({} when minimum is {})".format(len(src_pts),
                    #                                                                             MIN_MATCH_CURRENT))
                    end = True
            else:
                # print("Not enough matches remain: {}/{}".format(len(good_matches),
                #                                                 get_min_match_count_constant(area_window,
                #                                                                              area_test_image)))
                end = True
        else:
            # print("Discarded " + str(continuously_discarded_count) +
            #       " homography in a row. Not able to find other homographies")
            end = True
    return None, None, None, plots
