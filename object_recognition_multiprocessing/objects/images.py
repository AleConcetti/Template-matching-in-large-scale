import cv2



def rescale_image(image, n_pixels):
    height, width = image.shape[:2]
    if max([height, width]) > n_pixels:
        if height > width:
            image = cv2.resize(image,
                               (int(n_pixels * (width / height)), n_pixels),
                               interpolation=cv2.INTER_AREA)
        elif height < width:
            image = cv2.resize(image,
                               (n_pixels, int(n_pixels * (height / width))),
                               interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image,
                               (n_pixels, n_pixels),
                               interpolation=cv2.INTER_AREA)
    return image





class Image:
    def __init__(self, json, image_path):
        self.name = json['name']
        self.path = image_path + '/' + json['path']

        image = cv2.imread(self.path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image


class TestImage(Image):
    def __init__(self, json, image_path, resize=None):
        super().__init__(json, image_path)

        if resize is None:
            self.resize = json['resize']
        else:
            self.resize = resize
        self.scale_factor = self.get_scale_factor(self.image)
        self.image = rescale_image(self.image, self.resize)

    def get_scale_factor(self, image):
        n_pixels = self.resize
        height, width = image.shape[:2]
        if max([height, width]) > n_pixels:
            max_dim = max([height, width])
            return n_pixels / max_dim
        else:
            return 1

class TemplateImage(Image):
    def __init__(self, json, image_path, resize):
        super().__init__(json, image_path)

        self.resize = resize
        self.size = json['size']
        self.name= json['name']
        self.image = rescale_image(self.image, self.resize)
