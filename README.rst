===================================
FEniCS Error Estimation (FEniCS-EE)
===================================

Description
===========

FEniCS-EE is an open source library showing how various error estimation
strategies can be implemented in the FEniCS Project finite element solver
(https://fenicsproject.org). A particular focus is on implicit a posteriori
error estimators, which usually involve solving local error problems in special
finite element spaces on sub-patches of the mesh. Implicit error estimators
usually give a significantly sharper estimate of the exact error than the more
commonly used explicit error estimators, with very little extra computational
expense (small matrix-vector multiplications and linear solves).

FEniCS-EE is described in the paper:

** TODO: Add pre-print of paper **.

**FEniCS-EE is compatible with the 2019.2.0 development version of the FEniCS Project**.

Features
========

FEniCS-EE currently includes implementations of the following error
estimation techniques for the Poisson problem:

- Implicit residual estimator of Bank and Weiser,
- Implicit residual estimator of Verfürth,
- Recovery estimator of Zienkiewicz-Zhu,
- Explicit residual estimator of Babuška and Rheinbolt.

The following marking strategies:

- Maximum strategy,
- Dörfler strategy.

Upcoming features
=================

Please see the issue tracker for proposed features.

Getting started
===============

1. Install FEniCS by following the instructions at
   http://fenicsproject.org/download. We recommend using Docker to install
   FEniCS. However, you can use any method you want to install FEniCS.
2. Then, clone this repository using the command::

        git clone https://github.org/jhale/fenics-error-estimation

3. If you do not have an appropiate version of FEniCS already installed, use a Docker container 
   (skip the second line if you have already an appropiate version of FEniCS installed)::

        ./build-container.sh
        ./launch-container.sh

4. You should now have a shell inside a container with FEniCS installed.  Try
   out an example::

        python3 setup.py develop --user
        cd demo/pure_dirichlet
        python3 demo_pure-dirichlet.py

   The resulting fields are written to the directory ``output/`` which
   will be shared with the host machine. These files can be opened using
   `Paraview <http://www.paraview.org/>`_.

Automated testing
=================

We use CircleCI to perform automated testing. All documented demos include
basic sanity checks on the results. Tests are run in the
``rbulle/fenics-error-estimation`` Docker image.

FAQ
===

Q: Is this a replacement for the automated error estimation strategy already
   implemented in DOLFIN?

A: No, the examples in this repository are aimed at users who wish to implement
   specific a posteriori error estimation strategies into DOLFIN.

Q: Can you tackle goal-oriented mesh adaptivity problems?

A: Yes, see the demo `demo/goal_oriented_adaptivity/demo_goal-oriented-adaptivity.py`.
   We use a weighted marking strategy, as opposed to a weighted residual strategy,
   to control the error in the goal functional. This avoids solving the dual/adjoint
   problem in a higher-order space, or, ad-hoc extrapolation procedures.

Q: What will happen when FEniCSX
   https://fenicsproject.org/fenics-project-roadmap-2019/ is released?

A: The core routine of this package, a projected local assembler, will be
   reimplemented in Numba (http://numba.pydata.org/). The additional flexibility
   over the current C++ assembler will allow us to extend the package to various
   more complex problems, e.g. Stokes, incompressible elasticity,
   and singularly-perturbed reaction-diffusion equations.

Citing
======

Please consider citing the FEniCS-EE paper and code if you find it useful.

::
    @misc{bulle_fenics-ee_2019,
        title = {{FEniCS} {Error} {Estimation} {(FEniCS-EE)}},
        url = {https://figshare.com/articles/FEniCS-Shells/4291160},
        author = {Hale, Jack S. and Brunetti, Matteo and Bordas, Stéphane P.A. and Maurini, Corrado},
        month = dec,
        year = {2016},
        doi = {10.6084/m9.figshare.4291160},
        keywords = {FEniCS, error estimation},
    }

along with the appropriate general `FEniCS citations <http://fenicsproject.org/citing>`_.


Issues and Support
==================

Please use the issue tracker to report any issues.

For support or questions please email `jack.hale@uni.lu <mailto:jack.hale@uni.lu>`_.


Authors (alphabetical)
======================

| Raphaël Bulle, University of Luxembourg, Luxembourg.
| Jack S. Hale, University of Luxembourg, Luxembourg.

License
=======

FEniCS-EE is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with FEniCS-EE.  If not, see http://www.gnu.org/licenses/.