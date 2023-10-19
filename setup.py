# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shlex
import sys
import sysconfig
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

VERSION = "0.5.0.dev0"
URL = ""

REQUIREMENTS = [
    "numpy",
    "scipy",
    "fenics-dolfinx>=0.5.0<0.6.0",
]

AUTHORS = """\
Raphael Bulle
Jack S. Hale
"""

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Mathematics
"""


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: "
                               + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
        cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                       '-DPython3_EXECUTABLE=' + sys.executable,
                       f'-DPython3_LIBRARIES={sysconfig.get_config_var("LIBDEST")}',
                       f'-DPython3_INCLUDE_DIRS={sysconfig.get_config_var("INCLUDEPY")}']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        env = os.environ.copy()
        # default to 3 build threads
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in env:
            env["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"

        import pybind11
        env['pybind11_DIR'] = pybind11.get_cmake_dir()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)


def run_install():
    setup(name="fenicsx_error_estimation",
          description="Implicit a posteriori error estimators in FEniCSx",
          version=VERSION,
          author=AUTHORS,
          classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
          license="LGPL version 3 or later",
          author_email="mail@jackhale.co.uk",
          maintainer_email="mail@jackhale.co.uk",
          url=URL,
          packages=["fenicsx_error_estimation"],
          package_dir={"fenicsx_error_estimation": "fenicsx_error_estimation"},
          ext_modules=[CMakeExtension("fenicsx_error_estimation.cpp", sourcedir="./fenicsx_error_estimation/cpp/")],
          cmdclass=dict(build_ext=CMakeBuild),
          platforms=["Linux", "Mac OS-X", "Unix"],
          install_requires=REQUIREMENTS,
          setup_requires=["pybind11"],
          extras_require={"demos": ["pandas", "mpltools", "matplotlib"],
                          "lint": ["isort", "flake8"],
                          "test": ["pytest"],
                          "ci": ["fenicsx_error_estimation[demos]", "fenicsx_error_estimation[lint]", "fenicsx_error_estimation[test]"]},
          zip_safe=False)


if __name__ == "__main__":
    run_install()
