import time
import traceback
import multiprocessing

from functions.plot_manager import setup_backend_for_saving
from functions.process_functions import find_homographies_per_thread
from objects.constants import Constants
from objects.homography import Homography
from objects.images import TemplateImage, TestImage
from objects.plot_discarded import PlotDiscarded
from objects.ratio import RatioList


class ProcessHandler(multiprocessing.Process):

    def __init__(self, test_image: TestImage, template: TemplateImage, return_dict, ratio_list, plot_dict):
        super().__init__()
        self.test_image: TestImage = test_image
        self.template: TemplateImage = template
        self.homographies: [Homography] = []

        # ratio list condiviso fra i processi
        self.ratio_list = RatioList(self.test_image.image)
        # ratio list single process
        # self.ratio_list = RatioList(test_image.image)

        self.return_dict = return_dict
        self.plot_dict = plot_dict
        self.plots: [PlotDiscarded] = []

    def run(self):
        try:

            self.plot_dict[self.template.name] = []
            self.return_dict[self.template.name] = []
            tic = time.time()
            find_homographies_per_thread(self.template, self.test_image, self.ratio_list,
                                         self.homographies, self.plots)
            toc = time.time()
            t = round(toc - tic, 2)

            print("{} -> found: {}, discarded: {}, time: {}".format(self.template.name, len(self.homographies),
                                                                    len(self.plots), t))
            self.plot_dict[self.template.name] = self.plots
            self.return_dict[self.template.name] = self.homographies


        except:
            traceback.print_exc()
