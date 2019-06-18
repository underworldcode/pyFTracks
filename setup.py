from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


extensions = [Extension("ketcham", ["pyTracks/ketcham.pyx",
                                    "pyTracks/src/ketcham.c"],
                        include_dirs=["pyTracks/include"])]

setup(
  name='pyTracks',
  ext_modules=cythonize(extensions),
  include_dirs=[numpy.get_include()]
)
