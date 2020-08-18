from setuptools import setup, Extension
from os import path
import codecs


class get_numpy_include(object):
    """Returns Numpy's include path with lazy import"""

    def __str__(self):
        import numpy
        return numpy.get_include()

def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

extensions = [Extension("pyFTracks.annealing", ["pyFTracks/annealing.pyx"]),
              Extension("pyFTracks.thermal_history", ["pyFTracks/thermal_history.pyx"])]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pyFTracks',
    setup_requires=[
        'setuptools>=18.0',
        'numpy',
        'cython'
        ],
    version=get_version("pyFTracks/__init__.py"),
    description='Fission Track Modelling and Analysis with Python',
    ext_modules=get_version("pyFTracks/__init__.py"),
    include_package_data=True,
    include_dirs=[get_numpy_include()],
    long_description=long_description,
    url='https://github.com/rbeucher/pyFTracks.git',
    author='Romain Beucher',
    author_email='romain.beucher@unimelb.edu.au',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=["pyFTracks", "pyFTracks/ressources", "pyFTracks/radialplot"],
    keywords='geology thermochronology fission-tracks',
    install_requires=requirements,

)
