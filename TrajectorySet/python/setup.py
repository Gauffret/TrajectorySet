from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext = [
	Extension(
		"fisher",
		["fisher.pyx"],
		extra_compile_args=['-fopenmp'],
		extra_link_args=['-fopenmp']
	)
]

setup(
	name='fisher',
	ext_modules = cythonize(ext)
)