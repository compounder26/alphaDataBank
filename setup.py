from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "correlation_utils",
        ["correlation_utils.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="correlation_utils",
    ext_modules=cythonize(extensions, language_level=3, annotate=False)
)
