from objects.homography import Homography
from shapely.geometry import mapping
import json

from objects.images import TestImage


def generate_json_evaluation(total_homographies_found: [Homography], test_image: TestImage):
    test_image_name=test_image.name
    scale = test_image.scale_factor
    json_dict = {}

    for i, hom in enumerate(total_homographies_found):
        json_partial = {}

        hom: Homography = hom
        json_partial['label'] = hom.template.name

        m = mapping(hom.polygon)
        vertices = m["coordinates"][0]
        vertices = vertices[:-1]

        json_partial['points'] = [(round(float(v[0])/scale), round(float(v[1])/scale)) for v in vertices]

        json_dict[str(i)] = json_partial

    with open('performance_evaluation_{}.json'.format(test_image_name), 'w') as file_json:
        file_json.write(json.dumps(json_dict, indent=4))

    print("Json file for evaluation created")