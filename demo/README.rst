==========================
Brief description of demos
==========================

Ordering is from simplest to most complex.

Poisson
=======

``pure_dirichlet/demo_pure-dirichlet.py`` - Poisson problem, Bank-Weiser
estimator, pure Dirichlet boundary conditions. No adaptive mesh refinement.

``pure_neumann/demo_pure-neumann.py`` - Reaction-diffusion problem, Bank-Weiser
estimator, pure Neumann boundary conditions. No adaptive mesh refinement.

``adaptivity/demo_adaptivity.py`` - 2D L-shaped domain, Bank-Weiser estimator,
pure Dirichlet conditions, known analytical solution. Adaptive mesh refinement.

TODO for DOLFINx
================

``goal_oriented_adaptivity/demo_goal-oriented-adaptivity.py`` - 2D L-shaped
domain, Bank-Weiser estimator, pure Dirichlet conditions, goal functional, dual
problem, known analytical solution. Adaptive mesh refinement based on weighted
sum of estimators.

``three-dimensions/demo_three-dimensions.py`` - 3D corner problem, Bank-Weiser
estimator, pure Dirichlet conditions, adaptive mesh refinement.

Incompressible elasticity
=========================

``incompressible-elasticity/demo_incompressible-elasticity.py`` - Square Domain,
Khan-Powell-Silvester estimator, residual estimator, adaptive mesh refinement.

Stokes
======

``incompressible-elasticity/demo_incompressible-elasticity.py`` - Square Domain,
Liao-Silvester estimator, residual estimator, adaptive mesh refinement.
