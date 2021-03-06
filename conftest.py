import py
import sys
import os
from distutils.util import get_platform

ROOT = py.path.local(__file__).dirpath()
SETUP_PY = ROOT.join('setup.py')
BUILD = ROOT.join('build')

def pytest_configure(config):
    exe = sys.executable
    ROOT.chdir()
    ret = os.system('"%s" "%s" build' % (exe, SETUP_PY))
    if ret != 0:
        print "build failed, quitting py.test"
        raise KeyboardInterrupt
    plat_dir = "lib.%s-%s" % (get_platform(), sys.version[0:3])
    build_dir = BUILD.join(plat_dir)
    for so in build_dir.visit('*.so'):
        relname = so.relto(build_dir)
        so.copy(ROOT.join(relname))
