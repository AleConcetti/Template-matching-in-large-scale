from shapely.geometry import Polygon, Point

import numpy as np

from functions.plot_manager import save_discarded_homography
from objects.constants import Constants
from functions.rgb_histogram_matching import evaluete_homography
from objects.homography import Homography


class Ratio:
    def __init__(self, Hom, ratios, angles):
        self.H = Hom

        # ratio on side 0 in pixel/cm (left)
        self.r0 = ratios[0]

        # ratio on side 1 in pixel/cm (down)
        self.r1 = ratios[1]

        # ratio on side 2 in pixel/cm (right)
        self.r2 = ratios[2]

        # ratio on side 3 in pixel/cm (up)
        self.r3 = ratios[3]  # ratio on side 3 in pixel/cm (up)

        # angle 0 in degrees (up-left)
        self.angle0 = angles[0]

        # angle 1 in degrees (down-left)
        self.angle1 = angles[1]

        # angle 2 in degrees (down-right)
        self.angle2 = angles[2]

        # angle 3 in degrees (up-right)
        self.angle3 = angles[3]

    def getRatios(self):
        return self.r0, self.r1, self.r2, self.r3

    def getAngles(self):
        return self.angle0, self.angle1, self.angle2, self.angle3


class RatioList:

    def __init__(self, test_image):
        self.list: [Ratio] = []
        self.test_image_height, self.test_image_width = test_image.shape[0:2]
        # This are only for plotting, can be deleted in the future
        self.last_centroids_plotted: [Point] = []
        self.test_image = test_image
        # _________________________________________________________

    def __gaussian(self, x0=None, y0=None):
        """
        Create a 2d Gaussian over the image with center in x0, y0
        This gaussian is used to generate the weights of the weighted mean in order to estimate the ratio.
        In this way, each of all the homographies found in the image gives a contribution based on the distance
        from the point where the ratio is going to be estimated
        :param x0:
        :param y0:
        :return: a matrix
        """
        width = self.test_image_width
        height = self.test_image_height
        if x0 is None and y0 is None:
            x0 = width / 2
            y0 = height / 2

        sigma = 1

        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

        deltax = x0 / width * 2
        deltay = y0 / height * 2

        d = ((x + 1) - deltax) ** 2 + ((y + 1) - deltay) ** 2

        g = np.exp(-(d / (2.0 * sigma ** 2)))

        return g

    def __calculate_side_ratio(self, object_polygon, template):
        """
        This function, starting from the object polygon of the homography, calculate the actual ratios
        :param object_polygon: the polygon associated to the homography
        :return: 4 floats
        """
        # Coordinates of the verteces of the object polygon
        coords = object_polygon.exterior.coords

        # Sides is a list where all the sides' length are stored
        sides = []

        # Build sides list
        for i in range(len(coords) - 1):
            j = i + 1
            # Two consecutive points A and B
            A = coords[i]
            B = coords[j]

            # Coordinates of the two points
            xA = A[0]
            yA = A[1]
            xB = B[0]
            yB = B[1]

            # Side is the distance between them
            side = ((xA - xB) ** 2 + (yA - yB) ** 2) ** 0.5  # in pixel
            sides.append(side)

        side0, side1 = template.size
        side0 = float(side0)  # in cm (left)
        side1 = float(side1)  # in cm (down)
        side2 = float(side0)  # in cm (right)
        side3 = float(side1)  # in cm (up)

        new_r0 = sides[0] / side0
        new_r1 = sides[1] / side1
        new_r2 = sides[2] / side2
        new_r3 = sides[3] / side3

        return new_r0, new_r1, new_r2, new_r3

    def __sparse_representation_ratios(self):
        """
        The sparse representation consists in a matrix with the same dimension of the image where
        most of element are zeros. If I have found an homography with centroid in x0, y0 and I have
        estemated a ratio for side 0 of 23.234, there will be 23.234 in the matrix refering to side 0
        in position x0, y0
        :return: 4 matrix
        """
        ratio0_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))
        ratio1_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))
        ratio2_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))
        ratio3_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))
        for ratio in self.list:
            x, y = list(ratio.H.polygon.centroid.coords)[0]
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)

            r0, r1, r2, r3 = ratio.getRatios()
            ratio0_sparse_rep[y][x] = r0
            ratio1_sparse_rep[y][x] = r1
            ratio2_sparse_rep[y][x] = r2
            ratio3_sparse_rep[y][x] = r3
        return ratio0_sparse_rep, ratio1_sparse_rep, ratio2_sparse_rep, ratio3_sparse_rep

    def __sparse_representation_angles(self):
        """
        The sparse representation consists in a matrix with the same dimension of the image where
        most of element are zeros. If I have found an homography with centroid in x0, y0 and it has angle = 84,
        there will be 84 in the matrix refering to angle0 in position x0, y0
        :return: 4 matrix
        """
        angle0_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))
        angle1_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))
        angle2_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))
        angle3_sparse_rep = np.zeros((self.test_image_height, self.test_image_width))


        for ratio in self.list:
            x, y = list(ratio.H.polygon.centroid.coords)[0]
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)

            a0, a1, a2, a3 = ratio.getAngles()

            angle0_sparse_rep[y][x] = a0
            angle1_sparse_rep[y][x] = a1
            angle2_sparse_rep[y][x] = a2
            angle3_sparse_rep[y][x] = a3

        return angle0_sparse_rep, angle1_sparse_rep, angle2_sparse_rep, angle3_sparse_rep

    def __calculate_norm_dev(self, object_polygon, template):
        """
        Calculate the normalized deviation of the actual ratios from the expected ones
        :param object_polygon:
        :return: array of floats representing the deviation
        """
        # Build a sparse representation of the ratios
        ratio0_sparse_rep, \
        ratio1_sparse_rep, \
        ratio2_sparse_rep, \
        ratio3_sparse_rep = self.__sparse_representation_ratios()

        # Build a Gaussian over the image with center in the homography considered
        x, y = list(object_polygon.centroid.coords)[0]
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        gauss = self.__gaussian(x0=x, y0=y)

        # Pick the weights where the ratio is not zero
        weights = gauss[ratio0_sparse_rep != 0]
        sum_of_weights = sum(weights)

        # Weighted average of ratios
        expected_r0 = np.multiply(gauss, ratio0_sparse_rep).sum() / sum_of_weights
        expected_r1 = np.multiply(gauss, ratio1_sparse_rep).sum() / sum_of_weights
        expected_r2 = np.multiply(gauss, ratio2_sparse_rep).sum() / sum_of_weights
        expected_r3 = np.multiply(gauss, ratio3_sparse_rep).sum() / sum_of_weights

        # Calculate the ratio of the sides
        r0, r1, r2, r3 = self.__calculate_side_ratio(object_polygon, template)

        # Calculate the normalized deviations of each ratio based on the estimated ratio
        normalized_deviation = np.array([(r0 - expected_r0) / expected_r0,
                                         (r1 - expected_r1) / expected_r1,
                                         (r2 - expected_r2) / expected_r2,
                                         (r3 - expected_r3) / expected_r3])

        return normalized_deviation

    def __calculate_norm_dev_angles(self, object_polygon):
        """
        Calculate the normalized deviation of the actual angles from the expected ones
        :param object_polygon:
        :return: array of floats representing the deviation
        """
        # Build a sparse representation of the angles
        angle0_sparse_rep, \
        angle1_sparse_rep, \
        angle2_sparse_rep, \
        angle3_sparse_rep = self.__sparse_representation_angles()

        # Build a Gaussian over the image with center in the homography considered
        x, y = list(object_polygon.centroid.coords)[0]
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        gauss = self.__gaussian(x0=x, y0=y)

        # Pick the weights where the ratio is not zero
        weights = gauss[angle0_sparse_rep != 0]
        sum_of_weights = sum(weights)

        # Weighted average of angles
        expected_a0 = np.multiply(gauss, angle0_sparse_rep).sum() / sum_of_weights
        expected_a1 = np.multiply(gauss, angle1_sparse_rep).sum() / sum_of_weights
        expected_a2 = np.multiply(gauss, angle2_sparse_rep).sum() / sum_of_weights
        expected_a3 = np.multiply(gauss, angle3_sparse_rep).sum() / sum_of_weights

        # Calculate the ratio of the sides
        a0, a1, a2, a3 = self.__calculate_angles(object_polygon)

        # Calculate the normalized deviations of each ratio based on the estimated ratio
        normalized_deviation = np.array([(a0 - expected_a0) / expected_a0,
                                         (a1 - expected_a1) / expected_a1,
                                         (a2 - expected_a2) / expected_a2,
                                         (a3 - expected_a3) / expected_a3])

        return normalized_deviation

    def __root_mean_square_error(self, deviations):
        """
        This method just returns the RMS error
        :param normalized_deviations: the vector of deviations
        :return: a float
        """
        return ((deviations ** 2).sum() / len(deviations)) ** 0.5

    def __calculate_angles(self, polygon: Polygon):
        """
        This function calculates the angles of a polygon (4 sides)
        :param polygon:
        :return: 4 angles in degrees
        """
        # Coordinates of the verteces of the object polygon
        coords = polygon.exterior.coords

        # a______________d
        #  \            /|
        #   \          / |
        #    \        /  |
        #     \      /   |
        #      \    /    |
        #       \  /     |
        #        b/_____c|

        # Points
        a = Point(coords[0])
        b = Point(coords[1])
        c = Point(coords[2])
        d = Point(coords[3])

        angle_0 = self.__get_angle_of_triangle(b, a, d)
        angle_1 = self.__get_angle_of_triangle(a, b, d) + self.__get_angle_of_triangle(d, b, c)
        angle_2 = self.__get_angle_of_triangle(b, c, d)
        angle_3 = self.__get_angle_of_triangle(b, d, c) + self.__get_angle_of_triangle(b, d, a)

        return angle_0, angle_1, angle_2, angle_3

    def __get_angle_of_triangle(self, a: Point, b: Point, c: Point):
        """
        Given 3 points returns the angle laying in the middle point b.
        The Cosine Theorem has been leveraged:

        Cosine Theorem:
        Notation: angle_aBc is the angle with vertix in B of the triangle a-b-c
        ca^2 = ab^2 + bc^2 - 2*ab*bc*cos(angle_aBc)

        :param a: first point
        :param b: angle vertix
        :param c: second point
        :return: angle in degree
        """
        ab = a.distance(b)
        bc = b.distance(c)
        ca = c.distance(a)
        # ca^2 = ab^2 + bc^2 - 2*ab*bc*cos(angle_aBc)
        # ca^2 - ab^2 - bc^2 = - 2*ab*bc*cos(angle_aBc)
        # cos(angle_aBc) = (ab^2 + bc^2 - ca^2) / (2*ab*bc)
        # angle_aBc = arccos[(ab^2 + bc^2 - ca^2) / (2*ab*bc)]
        return np.degrees(np.arccos((ab ** 2 + bc ** 2 - ca ** 2) / (2 * ab * bc)))

    def add_new_ratio(self, Hom, template):
        # Calculate new ratios to add
        r0, r1, r2, r3 = self.__calculate_side_ratio(Hom.polygon, template)
        # print("\t{}: {}".format(template.name, ratios))
        # Calculate new angles to add
        angles = self.__calculate_angles(Hom.polygon)

        # Build the object Ratio
        ratio = Ratio(Hom, (r0, r1, r2, r3), angles)

        # Add the new object
        self.list.append(ratio)

    def is_homography_likely(self, object_polygon, template, test_image, idd,
                             big_window, id_hom_global, threshold=0.3):
        plot = None
        """
        This function estimate whether or not the polygon in input can be realistic to obserb in the sc ene or not
        :param plots:
        :param id_hom_global:
        :param big_window:
        :param idd:
        :param template: TemplateImage
        :param test_image:
        :param object_polygon:
        :param threshold:
        :return: A boolean. True if it is likely
        """
        # One sample is not enough to make a acceptable estimate of the ratio
        if self.list is None or len(self.list) < Constants.MIN_HOM_RATION:
            return True, plot

        # Calculate normalized deviation between real ratios and the estimated ones
        normalized_deviation = self.__calculate_norm_dev(object_polygon, template)
        normalized_deviation_angles = self.__calculate_norm_dev_angles(object_polygon)

        # All the ratio must be not too distant from the estimated ratio
        all_sides_likely = np.all(np.absolute(normalized_deviation) < threshold)

        # All the angles must be not too distant from the estimated angles
        all_angles_likely = np.all(np.absolute(normalized_deviation_angles) < threshold)

        # If the ratios are similar to the original one return True

        if all_sides_likely:
            if all_angles_likely:
                return True, plot
            else:
                if Constants.SAVE:
                    new_centroid = object_polygon.centroid
                    not_already_plotted = True
                    for centroid in self.last_centroids_plotted:
                        not_already_plotted = not_already_plotted and np.abs(new_centroid.distance(centroid)) > 0.1

                    if not_already_plotted:
                        self.last_centroids_plotted.append(new_centroid)
                        plot = save_discarded_homography(object_polygon, template.name, idd, "angles not likely", str(np.absolute(normalized_deviation_angles)))
                return False, plot
        else:
            # Ratios are not too similar,
            # so I have to check whether there's a scale trasformation or there's a wrong homography
            # A scale transformation is likely if the deviations are similar between each other

            # So I normalized again the array based on its mean
            mean = np.mean(normalized_deviation)
            bi_normalized_deviation = (normalized_deviation - mean) / mean

            # Then I check whether the difference between each other is not too big
            is_likely_to_be_scaled = np.all(np.absolute(bi_normalized_deviation) < threshold)

            if is_likely_to_be_scaled and all_angles_likely and False:
                # no angles check! It could be another objects with different angles
                return True, plot
            else:
                if Constants.SAVE:
                    new_centroid = object_polygon.centroid
                    not_already_plotted = True
                    for centroid in self.last_centroids_plotted:
                        not_already_plotted = not_already_plotted and np.abs(new_centroid.distance(centroid)) > 0.1

                    if not_already_plotted:
                        self.last_centroids_plotted.append(new_centroid)


                        plot = save_discarded_homography(object_polygon, template.name, idd, "sides not likely", str(np.absolute(normalized_deviation)))

                return False, plot


    # def evaluete_homography(self, h1, template):
        # object_polygon = h1.polygon
        # normalized_deviations = self.__calculate_norm_dev(object_polygon, template)
        # normalized_deviations_angles = self.__calculate_norm_dev_angles(object_polygon)
        # error_sides = self.__root_mean_square_error(normalized_deviations)
        # error_angles = self.__root_mean_square_error(normalized_deviations_angles)
        # error1 = max(error_sides, error_angles)
        # return error1





