import os

from objects.constants import Constants


def setup_path():
    """
    Setup the initial path
    :return:
    """
    if not os.path.exists(Constants.RESULT_PATH):
        os.mkdir(Constants.RESULT_PATH)

    path = Constants.SAVING_FOLDER_PATH

    if os.path.exists(path):
        _rename_path(path)

    os.mkdir(path)


def _rename_path(path):
    path_exist = True
    new_path = path
    i = 0
    while path_exist:
        i += 1
        new_path = path + ' ({})'.format(i)
        path_exist = os.path.exists(new_path)

    os.rename(path, new_path)
