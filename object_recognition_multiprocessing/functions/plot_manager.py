import os
from matplotlib import pyplot as plt
from shapely.geometry import Point
import cv2
import numpy as np

from objects.constants import Constants
from objects.homography import Homography
from objects.images import TestImage, TemplateImage
from objects.plot_discarded import PlotDiscarded

default_backend = None


def setup_backend():
    if os.uname()[0] == 'Linux':
        import matplotlib
        matplotlib.use('TkAgg')


def setup_backend_for_saving():
    global default_backend
    import matplotlib
    default_backend = matplotlib.get_backend()
    matplotlib.use('Agg')


def restore_backend():
    global default_backend

    import matplotlib
    matplotlib.use(default_backend)


def __get_path(name):
    n = name.lower().replace(' ', '_')
    path = Constants.SAVING_FOLDER_PATH + '/' + n

    if not os.path.exists(path):
        os.mkdir(path)

    return path


def save_homographies(test_image: TestImage, homographies: [Homography], template: TemplateImage = None,
                      before_overlap=False, before_filters=False):
    plt.clf()
    if before_filters:
        plt.imshow(test_image)
    else:
        plt.imshow(test_image.image)

    if template is not None:
        name = template.name.upper()
        title = 'Show all items found in template {}'.format(name)
        save = '/Show_items_found_in_template_{}.png'.format(name)
    elif before_overlap:
        title = 'Show all items found before overlap removing'
        save = '/Show_all_items_found_before_overlap_removing.png'
    elif before_filters:
        title = 'Show all items found before filters'
        save = '/Show_all_items_found_before_filters.png'
    else:
        title = 'Show all items found'
        save = '/Show_items_found.png'

    plt.title(title)
    for homography in homographies:
        x, y = homography.polygon.exterior.xy
        plt.plot(x, y, linewidth=2, color='r')

        x_c, y_c = homography.polygon.centroid.coords.xy
        x_c = round(x_c[0])
        y_c = round(y_c[0])
        # plt.text(x_c, y_c, homography.template.name + "-" + str(homography.id_hom_global), horizontalalignment='center',
        #          verticalalignment='center', fontsize=4, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(Constants.SAVING_FOLDER_PATH + save, dpi=400, format='png')


def save_homographies_for_template(test_image: TestImage, homographies: [Homography]):
    homographies_dict = {}
    for homography in homographies:
        name = homography.template.name
        if name not in homographies_dict:
            homographies_dict[name] = []

        homographies_dict[name].append(homography)

    for k, v in homographies_dict.items():
        for homography in v:
            plt.clf()
            plt.imshow(test_image.image)

            name = k.upper()
            iid = str(homography.id_hom_global)
            iid = iid.replace('.', '-')
            title = 'Show all items found in template after filters {}-{}'.format(name, iid)
            save = '/Show_items_found_in_template_{}_after_filters_{}.png'.format(name, iid)

            plt.title(title)
            x, y = homography.polygon.exterior.xy
            plt.plot(x, y, linewidth=2, color='r')

            x_c, y_c = homography.polygon.centroid.coords.xy
            x_c = round(x_c[0])
            y_c = round(y_c[0])
            plt.text(x_c, y_c, homography.template.name + "-" + str(homography.id_hom_global),
                     horizontalalignment='center',
                     verticalalignment='center', fontsize=4, bbox=dict(facecolor='white', alpha=0.5))

            plt.savefig(Constants.SAVING_FOLDER_PATH + save, dpi=400, format='png')


def save_homographies_report(test_image, homographies: [Homography]):
    plt.clf()
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=4)
    ax1 = fig.add_subplot(gs[:-1, 0])
    ax2 = fig.add_subplot(gs[-1, 0])
    ax3 = fig.add_subplot(gs[:, 1:])
    axs = [ax1, ax2, ax3]

    # fig, axs = plt.subplots(1, 3)

    for ax in axs:
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    for i, homography in enumerate(homographies):
        axs[i].imshow(homography.template.image)

        x, y = homography.polygon.exterior.xy
        if homography.template.name == 'CIOCCOLATO_FONDENTE':
            color = 'r'
            label = 'Chocolate'
        else:
            color = 'b'
            label = 'Special-k'
        axs[i].set_title(label)

        axs[2].imshow(test_image.image)
        axs[2].set_title('Items found')
        axs[2].plot(x, y, linewidth=2, color=color, label=label)
        axs[2].legend(loc='lower center')

    plt.savefig(Constants.SAVING_FOLDER_PATH + '/Show_all_items_report.png', dpi=400, format='png')


def save_keypoints_report(good_matches, test_keypoints, image, name):
    # test_image.image, good_matches, test_keypoints, hotpoints_image_after_elaboration, template.name
    # test, good_matches, keypoints, hotpoint, name
    keypoints_mask = np.ones(image.shape, dtype=np.uint8) * 255

    good_keypoints = []
    for good_match in good_matches:
        dst_pt = test_keypoints[good_match.queryIdx]
        good_keypoints.append(dst_pt)

    test_keypoints_image = cv2.drawKeypoints(keypoints_mask, good_keypoints, None)

    path = __get_path(name)
    plt.clf()
    fig, axs = plt.subplots(1, 2)

    for ax in axs:
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    # fig.suptitle('Hotpoints')
    axs[0].imshow(image)
    axs[1].imshow(test_keypoints_image, cmap='Greys')
    save = path + '/Keypoints_report.png'
    plt.savefig(save, dpi=400, format='png')


def save_hotpoints(image, name, again=False):
    path = __get_path(name)
    plt.clf()
    plt.imshow(image, cmap='Greys')
    if again:
        title = 'Hotpoints recomputed'
        save = path + '/Hotpoints_again.png'
    else:
        title = 'Hotpoints'
        save = path + '/Hotpoints.png'

    plt.title(title)
    plt.savefig(save, dpi=400, format='png')


def save_hotpoints_report(test, good_matches, keypoints, hotpoint, name):
    keypoints_mask = np.ones(test.shape, dtype=np.uint8) * 255

    good_keypoints = []
    for good_match in good_matches:
        dst_pt = keypoints[good_match.queryIdx]
        good_keypoints.append(dst_pt)

    test_keypoints_image = cv2.drawKeypoints(keypoints_mask, good_keypoints, None)

    path = __get_path(name)
    plt.clf()
    fig, axs = plt.subplots(1, 3)

    for ax in axs:
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    # fig.suptitle('Hotpoints')
    axs[0].imshow(test)
    axs[1].imshow(test_keypoints_image, cmap='Greys')
    axs[2].imshow(hotpoint, cmap='Greys')
    save = path + '/Hotpoints_report.png'
    plt.savefig(save, dpi=400, format='png')


def save_window(test_image, id_hotpoint, id_pos, id_homography, window, inliers_matches, test_keypoints, name):
    plt.clf()
    plt.imshow(test_image)
    plt.title("{}.{}.{} Window keypoints inside".format(id_hotpoint, id_pos, id_homography))
    # Plot window
    x, y = window.exterior.xy
    plt.plot(x, y, linewidth=1, color='g')
    # Plot inliers
    for i, match in enumerate(inliers_matches):
        point = Point(test_keypoints[match.queryIdx].pt)
        x, y = point.x, point.y
        if window.contains(point):
            plt.scatter(x, y, c='r', s=2, marker='x')
    path = __get_path(name)
    plt.savefig(path + "/{}.{}.{} (a) Window keypoints inside.png".format(id_hotpoint, id_pos, id_homography),
                dpi=400, format="png")


def save_two_polygons(test_image, id_hotpoint, id_pos, id_homography, object_polygon1, object_polygon2, name):
    plt.clf()
    plt.imshow(test_image)
    plt.title("{}.{}.{} Compare polygons".format(id_hotpoint, id_pos, id_homography))

    try:
        x, y = object_polygon1.exterior.xy
    except:
        x = []
        y = []
    plt.plot(x, y, linewidth=3, color='r')
    try:
        x, y = object_polygon2.exterior.xy
    except:
        x = []
        y = []
    plt.plot(x, y, linewidth=2, color='b')
    plt.legend(['BEFORE', 'AFTER'])

    path = __get_path(name)
    plt.savefig(path + "/{}.{}.{} (b) Compare polygons.png".format(id_hotpoint, id_pos, id_homography),
                dpi=400, format="png")


def save_inliers(test_image, id_hotpoint, id_pos, id_homography, object_polygon_found, inliers_found, test_keypoints,
                 name):
    plt.clf()
    plt.imshow(test_image)
    plt.title("{}.{}.{} Inliers".format(id_hotpoint, id_pos, id_homography))

    x, y = object_polygon_found.exterior.xy
    plt.plot(x, y, linewidth=3, color='g')
    for match in inliers_found:
        point = Point(test_keypoints[match.queryIdx].pt)
        x, y = point.x, point.y
        plt.scatter(x, y, c='r', s=2, marker='x')
    path = __get_path(name)
    plt.savefig(path + "/{}.{}.{} (c) Inliers.png".format(id_hotpoint, id_pos, id_homography),
                dpi=400, format="png")


def save_discarded_homography(object_polygon, name, idd, text_title, text_label):
    x, y = object_polygon.exterior.xy

    title = "HOMOGRAPHY DISCARDED " + text_title
    path = __get_path(name)

    return PlotDiscarded(x, y, title, path, idd, text_label)
