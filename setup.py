from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("tree_based_algos/tree_components/cython_support", sources=["tree_based_algos/tree_components/cython_support.pyx", "tree_based_algos/tree_components/c_background.c"]),
    Extension("tree_based_algos/pure_cython_tree_components/decision_node_cython", ["tree_based_algos/pure_cython_tree_components/decision_node_cython.pyx"]),
    ]

setup(
    ext_modules=cythonize(extensions)
)



