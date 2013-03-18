import os

INCLUDE = os.path.abspath(os.path.join((__file__), '..', '..', 'include'))

def get_numpy_include_dirs():
    return [INCLUDE]
