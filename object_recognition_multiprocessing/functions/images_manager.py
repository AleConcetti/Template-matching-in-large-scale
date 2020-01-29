import json
import sys
import traceback

from matplotlib import pyplot as plt

from objects.constants import Constants
from objects.images import TestImage, TemplateImage


def ask_test_image() -> TestImage:
    """
    Ask for the test image to use into the simulation
    :return: test image choosen
    :rtype: TestImage
    """

    TEST_IMAGE_PER_ROW = 4

    with open(Constants.TEST_JSON_PATH, 'r') as file:
        file_json = json.load(file)

        images_path = file_json['image_path']
        test_images_json = file_json['test_images']

        test_images = []
        for test_image in test_images_json:
            # print(test_image)
            test_images.append(TestImage(test_image, images_path))

        from math import ceil
        col = TEST_IMAGE_PER_ROW
        row = ceil(len(test_images) / col)

        fig = plt.figure(figsize=(10, 8))
        fig.suptitle("Choose the test image", fontsize=18, y=0.99)

        for i, test_image in enumerate(test_images):
            ax = fig.add_subplot(row, col, i + 1)
            plt.imshow(test_image.image)
            ax.set_title('id image: {}'.format(i), fontsize=10)
            plt.axis('off')
        plt.show()

        id_image = int(input('Digit the id of the target test image: '))

        try:
            test_image = test_images[id_image]

            plt.imshow(test_image.image)
            plt.title('Test image choosen')
            plt.axis('off')
            plt.show()

            return test_image
        except IndexError:
            print('No test image found')
            sys.exit(1)
        except Exception:
            traceback.print_exc()
            sys.exit(1)


def get_templates(template_path=None) -> [TemplateImage]:
    """
    Return all the templates
    :return: list of templates
    :rtype: [TemplateImage]
    """
    templates = []

    if template_path is None:
        template_path = Constants.TEMPLATE_JSON_PATH

    with open(template_path, 'r') as file:
        file_json = json.load(file)

        image_path = file_json['image_path']
        templates_json = file_json['templates']

        for template_json in templates_json:
            template = TemplateImage(template_json, image_path, Constants.MAX_PIXELS_TEMPLATE)
            templates.append(template)

    return templates
