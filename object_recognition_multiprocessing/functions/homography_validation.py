import numpy as np
from shapely.geometry import Point


def all_inliers_in_the_polygon(test_keypoints, inliers_matches, polygon):
    # Function that checks that all the inliers lie in the projected polygon

    # Retrieve coordinates of inliers keypoints in test image
    inlier_points = np.float32([test_keypoints[m.queryIdx].pt for m in inliers_matches])

    # For all the inliers check whether they are contained in the polygon, if not return False
    for inlier_point in inlier_points:
        point = Point(inlier_point)
        if not polygon.contains(point):
            return False

    return True


# Check if the polygon referred to the found object is not outside from the test image rectangle for a certain amount
def out_area_ratio(object_polygon, test_image_polygon, out_of_image_threshold):
    # Intersect object polygon and test image rectangle polygon
    object_polygon_in = test_image_polygon.intersection(object_polygon)

    # Compute the amount of area of the object polygon outside the image rectangle polygon
    area_out = object_polygon.area - object_polygon_in.area

    # Check the condition
    return area_out / object_polygon.area <= out_of_image_threshold
