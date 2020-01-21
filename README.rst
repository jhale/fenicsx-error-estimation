=======================
FEniCS Error Estimation
=======================

A FEniCS Project-based library for simulating thin structures.


Description
===========

FEniCS-Shells is an open-source library that provides finite element-based
numerical methods for solving a wide range of thin structural models (beams,
plates and shells) expressed in the Unified Form Language (UFL) of the `FEniCS
Project <http://fenicsproject.org>`_.

**FEniCS-Shells is compatible with the 2019.1.0 release of the FEniCS Project**.

FEniCS-Shells is described fully in the paper:

Simple and extensible plate and shell finite element models through automatic
code generation tools, J. S. Hale, M. Brunetti, S. P. A. Bordas, C. Maurini.
*Computers & Structures*, 209, 163-181, `doi:10.1016/j.compstruc.2018.08.001
<https://dx.doi.org/10.1016/j.compstruc.2018.08.001>`_.

Getting started
===============

1. Install FEniCS by following the instructions at
   http://fenicsproject.org/download. We recommend using Docker to install
   FEniCS. However, you can use any method you want to install FEniCS.
2. Then, clone this repository using the command::

        git clone https://bitbucket.org/unilucompmech/fenics-shells.git

3. If you do not have an appropiate version of FEniCS already installed, use a Docker container 
   (skip the second line if you have already an appropiate version of FEniCS installed)::
        
        cd fenics-shells
        ./launch-container.sh
        
4. You should now have a shell inside a container with FEniCS installed.  Try
   out an example::

        python3 setup.py develop --user
        cd demo
        ./generate_demos.py
        cd documented/reissner_mindlin_clamped
        python3 demo_reissner-mindlin-clamped.py

   The resulting fields are written to the directory ``output/`` which
   will be shared with the host machine. These files can be opened using
   `Paraview <http://www.paraview.org/>`_.

5. Check out the demos at https://fenics-shells.readthedocs.io/.

Documentation
=============

Documentation can be viewed at http://fenics-shells.readthedocs.org/.

.. image:: https://readthedocs.org/projects/fenics-shells/badge/
   :target: http://fenics-shells.readthedocs.org/
   :alt: Documentation Status

Automated testing
=================

We use Bitbucket Pipelines to perform automated testing. All documented demos
include basic sanity checks on the results. Tests are run in the
``quay.io/fenicsproject/stable:current`` Docker image.

Features
========

FEniCS-Shells currently includes implementations of the following structural
models:

- Kirchhoff-Love plates,
- Reissner-Mindlin plates,
- von-Kármán shallow shells,
- Reissner-Mindlin-von-Kármán shallow shells,
- non-linear and linear Naghdi shells with exact geometry.

Additionally, the following models are under active development:

- linear and non-linear Timoshenko beams,

We are using a variety of finite element numerical techniques including:

- MITC reduction operators,
- discontinuous Galerkin methods,
- reduced integration techniques.


Citing
======

Please consider citing the FEniCS-Shells paper and code if you find it useful.

::

    @article{hale_simple_2018,
        title = {Simple and extensible plate and shell finite element models through automatic code generation tools},
        volume = {209},
        issn = {0045-7949},
        url = {http://www.sciencedirect.com/science/article/pii/S0045794918306126},
        doi = {10.1016/j.compstruc.2018.08.001},
        journal = {Computers \& Structures},
        author = {Hale, Jack S. and Brunetti, Matteo and Bordas, Stéphane P. A. and Maurini, Corrado},
        month = oct,
        year = {2018},
        keywords = {Domain specific language, FEniCS, Finite element methods, Plates, Shells, Thin structures},
        pages = {163--181},
    }
    
    @misc{hale_fenics-shells_2016,
        title = {{FEniCS}-{Shells}},
        url = {https://figshare.com/articles/FEniCS-Shells/4291160},
        author = {Hale, Jack S. and Brunetti, Matteo and Bordas, Stéphane P.A. and Maurini, Corrado},
        month = dec,
        year = {2016},
        doi = {10.6084/m9.figshare.4291160},
        keywords = {FEniCS, Locking, MITC, PDEs, Python, Shells, thin structures},
    }

along with the appropriate general `FEniCS citations <http://fenicsproject.org/citing>`_.

Contributing
============

We are always looking for contributions and help with fenics-shells. If you
have ideas, nice applications or code contributions then we would be happy to
help you get them included. We ask you to follow the `FEniCS Project git
workflow <https://bitbucket.org/fenics-project/dolfin/wiki/Git%20cookbook%20for%20FEniCS%20developers>`_.


Issues and Support
==================

Please use the `bugtracker <http://bitbucket.org/unilucompmech/fenics-shells>`_
to report any issues.

For support or questions please email `jack.hale@uni.lu <mailto:jack.hale@uni.lu>`_.


Authors (alphabetical)
======================

| Matteo Brunetti, Université Pierre et Marie Curie, Paris.
| Jack S. Hale, University of Luxembourg, Luxembourg.
| Corrado Maurini, Université Pierre et Marie Curie, Paris.


License
=======

fenics-shells is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with fenics-shells.  If not, see http://www.gnu.org/licenses/.
