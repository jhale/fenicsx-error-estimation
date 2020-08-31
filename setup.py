# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import sys
import platform
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

on_rtd = os.environ.get('READTHEDOCS') == 'True'

VERSION = "2019.2.0.dev0"
URL = ""

if on_rtd:
    REQUIREMENTS = []
else:
    REQUIREMENTS = [
        "numpy",
        "fenics-ffc>=2019.2.0.dev0",
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
Operating System :: Microsoft :: Windows
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
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
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            if "CIRCLECI" in os.environ:
                build_args += ['--', '-j2']
            else:
                num_build_threads = max(1, multiprocessing.cpu_count() - 1)
                build_args += ['--', '-j' + str(num_build_threads)]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


def run_install():
    setup(name="fenics_error_estimation",
          description="Implicit and a posteriori error estimates in FEniCS",
          version=VERSION,
          author=AUTHORS,
          classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
          license="LGPL version 3 or later",
          author_email="mail@jackhale.co.uk",
          maintainer_email="mail@jackhale.co.uk",
          url=URL,
          packages=["fenics_error_estimation"],
          package_dir={"fenics_error_estimation": "fenics_error_estimation"},
          ext_modules=[CMakeExtension("fenics_error_estimation.cpp", sourcedir="./fenics_error_estimation/cpp/")],
          cmdclass=dict(build_ext=CMakeBuild),
          platforms=["Linux", "Solaris", "Mac OS-X", "Unix"],
          install_requires=REQUIREMENTS,
          zip_safe=False)


if __name__ == "__main__":
    run_install()
