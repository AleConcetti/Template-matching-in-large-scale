from shapely.geometry import Polygon

class Hotpoint:
    def __init__(self, contour):
        self.polygon = Polygon([(p[0][0], p[0][1]) for p in contour])
        self.area = self.polygon.area
        x, y = self.polygon.centroid.coords.xy
        self.x = round(x[0])
        self.y = round(y[0])

    def __str__(self):
        return "c: ({}, {}) - a: {}".format(self.x, self.y, self.area)

    def is_good(self):
        # optimize: add more clausule to define if hotpoints is good or not
        return True

    def __create_vertices(self, center_x, center_y, width, height):
        p1 = [center_x - width / 2, center_y - height / 2]
        p2 = [center_x + width / 2, center_y - height / 2]
        p3 = [center_x + width / 2, center_y + height / 2]
        p4 = [center_x - width / 2, center_y + height / 2]
        return p1, p2, p3, p4

    def generate_window(self, scale=3) -> Polygon:
        minx, miny, maxx, maxy = self.polygon.bounds
        dx = (maxx - minx)
        dy = (maxy - miny)

        height = dy * scale
        width = dx * scale
        p1 = [self.x - width / 2, self.y - height / 2]
        p2 = [self.x + width / 2, self.y - height / 2]
        p3 = [self.x + width / 2, self.y + height / 2]
        p4 = [self.x - width / 2, self.y + height / 2]

        window = Polygon([p1, p2, p3, p4])

        return window

    def generate_window_with_quadrant(self, width, height, quarter, scale=2):
        """
        :param width: first_obj_width of the rectangular containing the first homography found
        :param height: first_obj_height of the rectangular containing the first homography found
        :param quarter: represent the position of the with w.r.t. the object
        :param scale: scale factor for adaptive window
        :return: window
        """

        width = scale * width
        height = scale * height

        minx = self.x - width / 2
        maxx = self.x + width / 2
        miny = self.y - height / 2
        maxy = self.y + height / 2

        if quarter == 0:
            # 1-----------2
            # |   MASK    |
            # |   |----|  |
            # |   |obj_|  |
            # |           |
            # 4-----------3
            center_x = self.x
            center_y = self.y
            p1, p2, p3, p4 = self.__create_vertices(center_x, center_y, width, height)
        elif quarter == 1:
            # 1-----------2
            # |      | obj|
            # |      |----|
            # |           |
            # |  MASK     |
            # 4-----------3
            center_x = minx
            center_y = maxy
            p1, p2, p3, p4 = self.__create_vertices(center_x, center_y, width, height)
        elif quarter == 2:
            # 1----------2
            # |obj |      |
            # |----|      |
            # |           |
            # |   MASK    |
            # 4----------3-
            center_x = maxx
            center_y = maxy
            p1, p2, p3, p4 = self.__create_vertices(center_x, center_y, width, height)
        elif quarter == 3:
            # 1-----------2
            # |    MASK   |
            # |           |
            # |----|      |
            # |obj |      |
            # 4-----------3
            center_x = maxx
            center_y = miny
            p1, p2, p3, p4 = self.__create_vertices(center_x, center_y, width, height)
        elif quarter == 4:
            # 1-----------2
            # |   MASK    |
            # |           |
            # |      |----|
            # |      |obj |
            # 4-----------3
            center_x = minx
            center_y = miny
            p1, p2, p3, p4 = self.__create_vertices(center_x, center_y, width, height)
        else:
            raise ValueError('Quadrante must be between 0 and 4')

        window = Polygon([p1, p2, p3, p4])

        return window

    def generate_window_with_chessboard(self, width, height, position, scale=2):
        # -------------------------
        # |       |       |       |
        # |   1   |   2   |   3   |
        # |       |       |       |
        # -------------------------
        # |       |       |       |
        # |    8  |   0   |   4   |
        # |       |       |       |
        # -------------------------
        # |       |       |       |
        # |    7  |   6   |   5   |
        # |       |       |       |
        # -------------------------
        width = scale * width
        height = scale * height

        p1, p2, p3, p4 = self.__create_vertices(self.x, self.y, width, height)
        # 1-----------2
        # |           |
        # |     0     |
        # |           |
        # 4-----------3

        if position == 0:
            pass
        elif position == 1:
            x, y = p1
            p1 = [x - width, y - height]
            p2 = [x, y - height]
            p3 = [x, y]
            p4 = [x - width, y]
        elif position == 2:
            x, y = p1
            p1 = [x, y - height]
            p2 = [x + width, y - height]
            p3 = [x + width, y]
            p4 = [x, y]
        elif position == 3:
            x, y = p2
            p1 = [x, y - height]
            p2 = [x + width, y - height]
            p3 = [x + width, y]
            p4 = [x, y]
        elif position == 4:
            x, y = p2
            p1 = [x, y]
            p2 = [x + width, y]
            p3 = [x + width, y + height]
            p4 = [x, y + height]
        elif position == 5:
            x, y = p3
            p1 = [x, y]
            p2 = [x + width, y]
            p3 = [x + width, y + height]
            p4 = [x, y + height]
        elif position == 6:
            x, y = p3
            p1 = [x - width, y]
            p2 = [x, y]
            p3 = [x, y + height]
            p4 = [x - width, y + height]
        elif position == 7:
            x, y = p4
            p1 = [x - width, y]
            p2 = [x, y]
            p3 = [x, y + height]
            p4 = [x - width, y + height]
        elif position == 8:
            x, y = p4
            p1 = [x - width, y - height]
            p2 = [x, y - height]
            p3 = [x, y]
            p4 = [x - width, y]
        else:
            raise ValueError('Position must be between 0 and 8')

        window = Polygon([p1, p2, p3, p4])

        return window
