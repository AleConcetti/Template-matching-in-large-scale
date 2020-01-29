class Constants:
    """
    Class that contains statically all the variable
    """
    # PATH
    RESULT_PATH = './results'  # result folder
    SAVING_FOLDER_PATH = RESULT_PATH + '/last_iteration'  # last iteration folder (inside result)
    TEMPLATE_JSON_PATH = './templates.json'  # template images json file
    TEST_JSON_PATH = './test.json'  # test images json file

    # CONSTANTS PARAMETERS
    MAX_PIXELS_TEMPLATE = 500  # maximum number of pixels admitted in the template largest dimension
    MAX_PIXELS_TEST = 1000  # maximum number of pixels admitted in the test largest dimension
    LOWE_THRESHOLD = 0.8  # a match is kept only if the distance with the closest
    MIN_HOM_RATION = 1

    MAX_CONTINUOUSLY_DISCARDED = 5  # max number of homographies discarded in a row before stopping
    CONTINUOUSLY_DISCARDED_LOWER_BOUND = 5
    CONTINUOUSLY_DISCARDED_UPPER_BOUND = 30

    MIN_MATCH_COUNT = 30  # search for the template whether there are at least
    MIN_MATCH_COUNT_LOWERBOUND = 0
    MIN_MATCH_COUNT_UPPERBOUND = 30

    # Bounds max iters ransac
    MIN_MAX_ITERS = 100
    MAX_MAX_ITERS = 2000

    # validation homography
    MIN_MATCH_CURRENT = 5  # stop when your matched homography has less than that features
    RANSAC_REPROJECTION_ERROR = 5.0  # maximum allowed reprojection error to treat a point pair as an inlier
    OUT_OF_IMAGE_THRESHOLD = 0.1  # Homography kept only if the polygon is not

    THRESHOLD_HOTPOINT_AGAIN = 0.4  # threshold to compute againg find hotpoints
    WINDOW_SCALE = 1.3  # scaling applied to the window in order to search homographies

    # thresholds for computing the overlap
    OVERLAP_DISCARD_THRESHOLD = 0.8
    PARTIAL_OVERLAP_DISCARD_THRESHOLD = 0.7

    # SAVING
    SAVE = False

    def __init__(self):
        import warnings
        warnings.warn('The constant class contains all the constants of the algorithm, it must not be initialized')
