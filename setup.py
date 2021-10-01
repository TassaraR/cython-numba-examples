from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='conv_cy',
    ext_modules=cythonize("conv_cy.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)
