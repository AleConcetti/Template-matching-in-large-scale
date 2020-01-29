from shapely.ops import cascaded_union

from objects.constants import Constants
from objects.homography import Homography


class Overlap:
    def __init__(self, h1: Homography, h2: Homography):
        self.h1: Homography = h1
        self.h2: Homography = h2

        self.p1 = []
        self.p2 = []
        self.p_common = []

        self.a1 = None
        self.a2 = None
        self.a_overlap = None
        self.merge_polygon = None
        self.a_merge = None
        self.overlap = None

    def compute_area(self):

        self.a1 = self.h1.polygon.area
        self.a2 = self.h2.polygon.area

        self.merge_polygon = cascaded_union([self.h1.polygon, self.h2.polygon])

        self.a_merge = self.merge_polygon.area

        self.a_overlap = self.a1 + self.a2 - self.a_merge

        self.overlap = self.a_overlap / ((self.a1 + self.a2) / 2)

    def is_duplicate(self):
        if self.is_contained():
            return True

        self.compute_area()

        # print(self.get_report())

        if self.remove_partially_contained() is not None:
            return True

        if round(self.overlap, 2) > Constants.OVERLAP_DISCARD_THRESHOLD:
            return True

        return False

    def duplicate_to_discard(self):
        if self.h1.polygon.contains(self.h2.polygon):
            print('Remove H2 (for contains)')
            return self.h2
        if self.h2.polygon.contains(self.h1.polygon):
            print('Remove H1 (for contains)')
            return self.h1

        if round(self.overlap, 2) > Constants.OVERLAP_DISCARD_THRESHOLD:
            print('Remove H2 (for overlap)')
            return self.h2

        remove = self.remove_partially_contained(to_discard=True)

        if remove is None:
            print('Remove H2 (why not)')
            return self.h2
        else:
            return remove

    # def check_point(self, point: Point):
    #     if self.h1.polygon.contains(point):
    #         self.p1.append(point)
    #
    #     if self.h2.polygon.contains(point):
    #         self.p2.append(point)
    #
    #     if self.h1.polygon.contains(point) and self.h2.polygon.contains(point):
    #         self.p_common.append(point)

    def get_report(self):
        if self.overlap is None or self.overlap == 0:
            self.compute_area()

        return "overlap: {} \t a1: {}, a2: {}, common area: {}".format(round(self.overlap, 2), self.a1, self.a2,
                                                                       self.a_overlap)

    def is_contained(self):
        p1, p2 = self.h1.polygon, self.h2.polygon
        if p1.contains(p2):
            print('H1 contains H2')
            return True
        if p2.contains(p1):
            print('H2 contains H1')
            return True

        return False

    def contain_to_remove(self, h: Homography):
        if self.h1 == h:
            return True
        if self.h2 == h:
            return True

        return False

    def remove_partially_contained(self, text=False, to_discard=False):
        if text:
            print("A1: {}\tA2: {}".format(round(self.a_overlap / self.a1, 2), round(self.a_overlap / self.a2, 2)))

        if self.a_overlap / self.a1 > Constants.PARTIAL_OVERLAP_DISCARD_THRESHOLD:
            if to_discard:
                print('Remove H1 (for partial overlap)')
            return self.h1

        if self.a_overlap / self.a2 > Constants.PARTIAL_OVERLAP_DISCARD_THRESHOLD:
            if to_discard:
                print('Remove H2 (for partial overlap)')
            return self.h2

        return None
