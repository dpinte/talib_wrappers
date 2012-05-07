""" setup.py for the talib wrappers.

You will need a working installation of TA-lib. You can find the library here::

    http://ta-lib.org/

Update the path to the include and library directories to build the wrappers.

"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import sys

if sys.platform == "linux2" :
    TALIB_INCLUDE_DIR = "/usr/local/include/ta-lib/"
    TALIB_LIB_DIR = "/usr/local/lib/"
elif sys.platform == "win32":
    TALIB_INCLUDE_DIR = r"c:\msys\1.0\local\include\ta-lib"
    TALIB_LIB_DIR = r"c:\msys\1.0\local\lib"
elif sys.platform =='darwin':
    TALIB_INCLUDE_DIR = "/usr/local/include/ta-lib/"
    TALIB_LIB_DIR = "/usr/local/lib/"


abstract_extension = Extension(
    "ta_abstract", 
    ["ta_abstract.pyx"],
    include_dirs = [
        numpy.get_include(),
        TALIB_INCLUDE_DIR]
    ,
    library_dirs = [TALIB_LIB_DIR],
    libraries = ["ta_lib"]
)

setup(
    ext_modules=[abstract_extension],
    cmdclass = {'build_ext': build_ext}
)
