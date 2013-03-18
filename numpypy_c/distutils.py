import os

INCLUDE = os.path.abspath(os.path.join((__file__), '..', '..', 'include'))

def get_include():
    return INCLUDE
