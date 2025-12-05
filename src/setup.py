from setuptools import setup
from Cython.Build import cythonize

# Build fastpaths.py into a C extension for speed.
setup(
    name="theta-fastpaths",
    ext_modules=cythonize(
        ["fastpaths.py", "fastpaths_vm.pyx"],
        compiler_directives={"language_level": "3"}
    ),
)
