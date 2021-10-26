===================================
FEniCSx Error Estimation (FEniCSx-EE)
===================================

Description
===========

FEniCSx-EE is an open source library showing how various error estimation
strategies can be implemented in the FEniCSx Project finite element solver
(https://fenicsproject.org). A particular focus is on implicit hierarchical a
posteriori error estimators, that usually involve solving local error problems
in special finite element spaces on cells of the mesh.

FEniCSx-EE is described in the pre-print:

Hierarchical a posteriori error estimation of Bank-Weiser type in the FEniCS
Project, R. Bulle, J. S. Hale, A. Lozinski, S. P. A. Bordas, F. Chouly,
(https://arxiv.org/abs/2102.04360).

**FEniCS-EE is compatible with the development version of the FEniCSx
Project (https://github.com/FEniCS)**.

**A version for FEniCS 2019.1.0 is available at (https://github.com/rbulle/fenics-error-estimation)**.

Features
========

FEniCS-EE currently includes implementations of the following error
estimation techniques for the Poisson problem:

- Implicit residual estimator of Bank and Weiser.

Upcoming features
=================

- Implicit residual estimator of Verfürth,

the following error estimation techniques for the incompressible
elasticity problem:

- Implicit residual estimator of Khan, Powell and Silvester (https://arxiv.org/abs/1710.03328),

and the following error estimation techniques for the Stokes
problem:

- Implicit residual estimator of Liao and Silvester (https://doi.org/10.1016/j.apnum.2010.05.003).

The following marking strategies:

- Maximum (bulk),
- Dörfler (equilibration).

Getting started
===============

1. Then, clone this repository using the command::

        git clone https://github.org/jhale/fenicsx-error-estimation

2. We currently require a custom build of FEniCSx::

        cd docker
        ./build-images.sh
        cd ../
        ./launch-container.sh

3. You should now have a shell inside a container with FEniCS installed.  Try
   out an example::

        python3 setup.py install
        cd demo/pure_dirichlet
        python3 demo_pure-dirichlet.py

   The resulting fields are written to the directory ``output/`` which
   will be shared with the host machine. These files can be opened using
   `Paraview <http://www.paraview.org/>`_.

Automated testing
=================

We use GitHub Actions to perform automated testing. All documented demos include
basic sanity checks on the results.

FAQ
===

TODO

Citing
======

Please consider citing the FEniCS-EE paper and code if you find it useful.

.. code::

  @misc{bulle2021hierarchical,
      title={Hierarchical a posteriori error estimation of Bank-Weiser type in the FEniCS Project}, 
      author={Raphaël Bulle and Jack S. Hale and Alexei Lozinski and Stéphane P. A. Bordas and Franz Chouly},
      year={2021},
      eprint={2102.04360},
      archivePrefix={arXiv},
      primaryClass={math.NA}
  }

  @misc{bulle_fenics-ee_2019,
        title = {{FEniCS} {Error} {Estimation} {(FEniCS-EE)}},
        author = {Bulle, Raphaël, and Hale, Jack S.},
        month = jan,
        year = {2019},
        doi = {10.6084/m9.figshare.10732421},
        keywords = {FEniCS, finite element methods, error estimation},
  }

along with the appropriate general `FEniCS citations <http://fenicsproject.org/citing>`_.


Issues and Support
==================

Please use the issue tracker to report any issues.


Authors (alphabetical)
======================

| Raphaël Bulle, University of Luxembourg, Luxembourg.
| Jack S. Hale, University of Luxembourg, Luxembourg.

License
=======

FEniCSx-EE is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with FEniCSx-EE.  If not, see http://www.gnu.org/licenses/.
