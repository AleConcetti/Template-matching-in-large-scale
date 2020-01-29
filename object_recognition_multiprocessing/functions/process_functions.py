import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from functions.find_homographies_double_check import find_homography_double_check
from functions.morphological_transformation import morph_transformation
from functions.plot_manager import save_hotpoints, save_hotpoints_report, save_keypoints_report
from objects.constants import Constants
from objects.homography import Homography
from objects.hotpoint import Hotpoint
from objects.images import TemplateImage, TestImage
from objects.overlap import Overlap
from objects.ratio import RatioList
from functions.rgb_histogram_matching import best_homography


class StopProcess(Exception):
    pass


def find_keypoints_and_good_matches(template, test_image):
    # Extract keypoints from the images with SIFT
    sift_detector = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with the SIFT detector from template and test images
    template_keypoints, template_descriptors = sift_detector.detectAndCompute(template.image, None)
    test_keypoints, test_descriptors = sift_detector.detectAndCompute(test_image, None)

    # Specify a constant representing the type of algorithm used by FLANN matcher
    flann_matcher_INDEX_KDTREE = 1  # algorithm used is KDTREE

    # Specify FLANN matcher constructor parameters
    index_params = dict(algorithm=flann_matcher_INDEX_KDTREE, trees=5)  # 5 trees used in the KDTREE search
    search_params = dict(checks=50)  # number of times the trees in the index should be recursively traversed

    # Create FLANN matcher
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Invoke FLANN matcher matching method: for each keypoint descriptor in the test image returns the k closest
    # #descriptors in the template image
    matches = flann_matcher.knnMatch(test_descriptors,
                                     template_descriptors,
                                     k=2)  # there is no trehsold, the k closest points are returned

    # Apply Lowe's ratio test to the found matches
    # Store all the good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < Constants.LOWE_THRESHOLD * n.distance:
            good_matches.append(m)

    return good_matches, test_keypoints, template_keypoints


def find_hotpoints(test_image, good_matches, test_keypoints):
    hotpoints_image = np.zeros(test_image.shape[:2])

    for match in good_matches:
        point = Point(test_keypoints[match.queryIdx].pt)
        x, y = point.x, point.y
        x = round(x)
        y = round(y)
        hotpoints_image[y][x] = 1

    # morphological transformation in order to find regions containing hotpoints
    hotpoints_image_after_elaboration = morph_transformation(hotpoints_image)

    # edge detection
    edged = hotpoints_image_after_elaboration.astype(np.uint8)
    edged *= 255
    edged = cv2.Canny(edged, 30, 200)

    # contours detection
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    good_hotpoints = []

    frame = np.zeros(hotpoints_image_after_elaboration.shape)
    maxy = frame.shape[0]
    maxx = frame.shape[1]
    # for cycle to filter and draw good hotpoints
    for contour in contours:
        if len(contour) > 2:
            hotpoint = Hotpoint(contour)

            if hotpoint.is_good():
                good_hotpoints.append(hotpoint)

            # prepare image with hotpoints region and center
            cv2.drawContours(frame, contour, -1, (255, 255, 0), 2)

            for id_hotpoint in range(-3, 3):
                for j in range(-3, 3):
                    y = min(hotpoint.y + id_hotpoint, maxy - 1)
                    x = min(hotpoint.x + j, maxx - 1)
                    frame[y][x] = 255

    return hotpoints_image_after_elaboration, hotpoints_image, good_hotpoints


def remove_overlaps(homographies, who_is_the_best):
    """
    :param homographies: list of homographies
    :param who_is_the_best: function that confronts 2 homographies and returns the best
    :return:
    """
    overlaps = []
    homographies_to_remove = []

    for i, h1 in enumerate(homographies):
        for j, h2 in enumerate(homographies[i + 1:]):
            if h1.polygon.overlaps(h2.polygon):
                overlaps.append(Overlap(h1, h2))

    # i = 0
    # while i < len(overlaps):
    for i in range(len(overlaps)):
        try:
            overlap = overlaps[i]

            if overlap.is_duplicate():

                best = who_is_the_best(overlap.h1, overlap.h2)
                if best == overlap.h1:
                    to_remove = overlap.h2
                elif best == overlap.h2:
                    to_remove = overlap.h1
                else:
                    print("Error in remove overlap!")
                    raise Exception
                # to_remove = overlap.duplicate_to_discard()

                homographies_to_remove.append(to_remove)

                for j, over in enumerate(overlaps[i + 1:]):
                    if over.contain_to_remove(to_remove):
                        overlaps.remove(over)

        except Exception:
            pass

    return homographies_to_remove


def find_homographies_per_thread(template: TemplateImage, test_image: TestImage,
                                 ratio_list: RatioList, homographies: [Homography],
                                 discarded_plots):
    good_matches, test_keypoints, template_keypoints = find_keypoints_and_good_matches(template, test_image.image)
    if Constants.SAVE:
        save_keypoints_report(good_matches, test_keypoints, test_image.image, template.name)

    # =================================================
    #           TROVARE GLI HOT POINTS
    # =================================================
    hotpoints_image_after_elaboration, hotpoints_image, good_hotpoints = find_hotpoints(test_image.image,
                                                                                        good_matches, test_keypoints)
    if Constants.SAVE:
        save_hotpoints(hotpoints_image_after_elaboration, template.name)
        save_hotpoints_report(test_image.image, good_matches, test_keypoints, hotpoints_image_after_elaboration, template.name)

    starting_matches = len(good_matches)
    id_homography = 0
    id_hom_global = [0]

    if len(good_hotpoints) <= 0:
        return

    # order good contours depending on the size of area
    good_hotpoints = sorted(good_hotpoints, key=lambda x: x.area, reverse=True)

    # =================================================
    #                    PRIMA ISTANZA
    # =================================================

    first_hotpoint = good_hotpoints[0]
    window = first_hotpoint.generate_window()

    homography, good_matches, _, plots = find_homography_double_check(test_image.image,
                                                                      template, good_matches,
                                                                      window,
                                                                      test_keypoints, template_keypoints,
                                                                      discarded_plots,
                                                                      id_hom_global=id_hom_global,
                                                                      id_hotpoint=0, id_pos=0,
                                                                      id_homography=id_homography,
                                                                      ratios_list=ratio_list)
    discarded_plots = plots

    if homography is not None:
        id_homography += 1
        homographies.append(homography)

        # =================================================
        #           COMPUTE AGAIN HOTPOINT
        # =================================================
        # height, width, _ = test_image.get_image().shape  # 1000, 750, 3
        # test_image_polygon = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        remaining_matches = len(good_matches)
        matches_used = 1 - (remaining_matches / starting_matches)

        # optimize: qui si può migliorare aggiungendo altri fattori
        if matches_used > Constants.THRESHOLD_HOTPOINT_AGAIN:
            hotpoints_image_after_elaboration, hotpoints_image, good_hotpoints = find_hotpoints(test_image.image,
                                                                                                good_matches,
                                                                                                test_keypoints)

            if len(good_hotpoints) == 0:
                raise StopProcess

                # order good contours depending on the size of area
            good_hotpoints = sorted(good_hotpoints, key=lambda x: x.area, reverse=True)

            if Constants.SAVE:
                save_hotpoints(hotpoints_image_after_elaboration, template.name, again=True)

        # =================================================
        #           CICLA SU TUTTI GLI HOTPOINTS
        # =================================================

        first_obj_width = homographies[0].width
        first_obj_height = homographies[0].height
        for id_hotpoint, hotpoint in enumerate(good_hotpoints):
            # Analyze the neighbourhood of the hotpoint with chessboard 3x3
            max_position = 9

            for position in range(max_position):
                window = hotpoint.generate_window_with_chessboard(width=first_obj_width,
                                                                  height=first_obj_height,
                                                                  position=position, scale=Constants.WINDOW_SCALE)

                searching = True
                while searching:
                    homography, good_matches, _, plots = find_homography_double_check(test_image.image,
                                                                                      template,
                                                                                      good_matches, window,
                                                                                      test_keypoints,
                                                                                      template_keypoints,
                                                                                      discarded_plots,
                                                                                      id_hom_global=id_hom_global,
                                                                                      id_hotpoint=id_hotpoint,
                                                                                      id_pos=position,
                                                                                      id_homography=id_homography,
                                                                                      ratios_list=ratio_list)
                    discarded_plots = plots

                    if homography is not None:
                        id_homography += 1
                        homographies.append(homography)

                    else:
                        searching = False

    # =================================================
    #               BIG WINDOW SEARCH
    # =================================================
    height, width, _ = test_image.image.shape  # 1000, 750, 3
    window = Polygon([(0, 0), (width, 0), (width, height), (0, height)])

    searching = True
    while searching:
        homography, good_matches, _, plots = find_homography_double_check(test_image.image,
                                                                          template,
                                                                          good_matches, window,
                                                                          test_keypoints, template_keypoints,
                                                                          discarded_plots,
                                                                          id_hom_global=id_hom_global,
                                                                          id_hotpoint=-1,
                                                                          id_pos=-1,
                                                                          id_homography=id_homography,
                                                                          ratios_list=ratio_list,
                                                                          big_window=True)
        discarded_plots = plots

        if homography is not None:
            id_homography += 1
            homographies.append(homography)

        else:
            searching = False

    # =================================================
    #            FIND HOMOGRAPHIES OVERLAP
    # =================================================

    to_removes = remove_overlaps(homographies, lambda h1, h2: best_homography(h1, h2, test_image.image))

    for to_remove in to_removes:
        homographies.remove(to_remove)
