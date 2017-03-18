from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("tree_based_algos/tree_components/cython_support", ["tree_based_algos/tree_components/cython_support.pyx"])
    ]

setup(
    ext_modules=cythonize(extensions)
)

