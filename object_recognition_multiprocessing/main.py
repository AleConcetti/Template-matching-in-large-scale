import multiprocessing
import time

from functions.final_filters import final_filter
from functions.images_manager import ask_test_image, get_templates
from functions.path_manager import setup_path
from functions.plot_manager import setup_backend, setup_backend_for_saving, \
    restore_backend, save_homographies
from objects.process import ProcessHandler
from objects.ratio import RatioList
from objects.ratio_manager import RatioManager

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # check if result path exists otherwise they are created
    setup_path()

    # setup backend of matplotlib for linux users
    setup_backend()

    # choose test image
    test_image = ask_test_image()

    # get all template
    templates = get_templates()

    # list of processes
    template_processes: [ProcessHandler] = []

    setup_backend_for_saving()

    # templates = [templates[7]]
    print("Number of templates: {}".format(len(templates)))

    manager = multiprocessing.Manager()
    # create the dict to return the homography
    return_dict = manager.dict()
    # create the list to plot and save all homographies
    plot_dict = manager.dict()

    # create and register a ratio
    RatioManager.register("ratio_list", RatioList)
    ratio_manager = RatioManager()
    ratio_manager.start()
    ratio_list = ratio_manager.ratio_list(test_image.image)

    tic = time.time()

    for template in templates:
        process = ProcessHandler(test_image, template, return_dict, ratio_list, plot_dict)
        if process is not None:
            template_processes.append(process)
            process.start()

    for process in template_processes:
        process.join()

    toc = time.time()

    restore_backend()

    print('-' * 100)

    # list of total homographies found and number of homographies discarded
    total_homographies_found = []
    homographies_discarded = 0

    for process in template_processes:
        print(process.template.name, end='')
        try:
            homographies = return_dict[process.template.name]
            total_homographies_found += homographies

            if len(return_dict[process.template.name]) > 0:
                save_homographies(process.test_image, return_dict[process.template.name], process.template)

            plots = plot_dict[process.template.name]

            # print("Number of items found in {}: {}".format(process.template.name, len(return_dict[process.template.name])))
            # print("Number of items discarded  in {}: {}".format(process.template.name, len(plots)))

            homographies_discarded += len(plots)
            for plot in plots:
                plot.save_plot(test_image.image)
            print(', completed!')

        except:
            print("\nERROR:")
            print(process.template.name + " not in dictionary")
            print(return_dict.keys())
            print("*" * 20)

    before_overlaps = len(total_homographies_found)
    save_homographies(test_image, total_homographies_found, before_overlap=True)

    #save_homographies_report(test_image, total_homographies_found)

    total_homographies_found = final_filter(total_homographies_found, test_image.image)
    # save_homographies_for_template(test_image, total_homographies_found)

    # generate_json_evaluation(total_homographies_found, test_image)

    save_homographies(test_image, total_homographies_found)

    print('=' * 100)
    print("Computational time: {}".format(round(toc - tic, 2)))
    print("Number of total items found: {}".format(len(total_homographies_found)))
    print("Number of total items found before overlap: {}".format(before_overlaps))
    print("Number of total items discarded: {}".format(homographies_discarded))
    print('=' * 100)

    """
    # plot all the items found in the test image
    from matplotlib import pyplot as plt
    import time

    plt.clf()
    plt.imshow(test_image.image)
    plt.title('Animation')
    input('Sta per partire l\'animazione')
    time.sleep(5)
    plt.figure(1)
    for i in range(len(total_homographies_found)):
        h = total_homographies_found[i]
        plt.imshow(test_image.image)
        # plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.title('Items found {}.{}.{}'.format(h.id_hotpoint, h.id_pos, h.id_homography))
        for j in range(i + 1):
            homography = total_homographies_found[j]
            x, y = homography.polygon.exterior.xy
            if i == j:
                plt.plot(x, y, c='r')
            else:
                plt.plot(x, y, c='b')
        plt.pause(0.1)
        plt.gca().clear()
    """

