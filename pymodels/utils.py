from __future__ import division, print_function

import os
import errno


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def ensure_parent_exists(path):
    mkdir(os.path.dirname(path))


def path_rel_to_base(*components):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, *components
    ))
