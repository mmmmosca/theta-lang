from setuptools import setup
from Cython.Build import cythonize
from pathlib import Path

# Build fastpaths.py into a C extension for speed.
base = Path(__file__).parent.resolve()

setup(
    name="theta-fastpaths",
    ext_modules=cythonize(
        [str(base / "fastpaths.py"), str(base / "fastpaths_vm.pyx")],
        compiler_directives={"language_level": "3"}
    ),
)
