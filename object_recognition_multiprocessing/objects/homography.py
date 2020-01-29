import time

from shapely.geometry import Polygon

from objects.images import TemplateImage


class Homography:
    def __init__(self, H, polygon: Polygon, id_hotpoint: int, id_pos: int, id_homography: int, id_hom_global: int,
                 template: TemplateImage, num_inliers: int):

        self.H = H
        self.polygon: Polygon = polygon
        self.template: TemplateImage = template

        self.id_hotpoint = id_hotpoint
        self.id_pos = id_pos
        self.id_homography = id_homography
        self.id_hom_global = time.time()

        minx, miny, maxx, maxy = self.polygon.bounds
        self.width = (maxx - minx)
        self.height = (maxy - miny)

        self.timestamp = time.time()
        self.num_inliers = num_inliers

    def get_id(self):
        return "<{}, {}, {}>".format(self.id_hotpoint, self.id_pos, self.id_homography)