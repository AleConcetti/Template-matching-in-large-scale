import os

from matplotlib import pyplot as plt


class PlotDiscarded:
    def __init__(self, x, y, title, save_path, idd, text):
        self.x = x
        self.y = y
        self.title = title
        self.path = save_path
        self.idd = idd
        self.text = text

    def __str__(self):
        return self.title

    def save_plot(self, test_image):
        plt.clf()
        plt.imshow(test_image)

        plt.title(self.title)

        plt.plot(self.x, self.y, linewidth=2, color='r')
        plt.text(100,100, self.text, bbox=dict(facecolor='white', alpha=0.8), fontsize=7)

        save_path = self.path + "/homography discarded {}".format(self.idd)
        if os.path.exists(save_path + '.png'):
            items = 1
            while os.path.exists(save_path + ' ({}).png'.format(items)):
                items += 1
            save_path = save_path + ' ({})'.format(items)

        save_path = save_path + '.png'

        plt.savefig(save_path, dpi=400, format="png")
