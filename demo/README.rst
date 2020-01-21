==========================
Brief description of demos
==========================

Ordering is from simplest to most complex.

``pure_dirichlet/demo_pure-dirichlet.py`` - Poisson problem, Bank-Weiser
estimator, pure Dirichlet boundary conditions. No adaptive mesh refinement.

``pure_neumann/demo_pure-neumann.py`` - Reaction-diffusion problem, Bank-Weiser
estimator, pure Neumann boundary conditions. No adaptive mesh refinement.

``adaptivity/demo_adaptivity.py`` - 2D L-shaped domain, Bank-Weiser estimator,
pure Dirichlet conditions, known analytical solution. Adaptive mesh refinement.

``goal_oriented_adaptivity/demo_goal-oriented-adaptivity.py`` - 2D L-shaped
domain, Bank-Weiser estimator, pure Dirichlet conditions, goal functional, dual
problem, known analytical solution. Adaptive mesh refinement based on weighted
sum of estimators.

``three-dimensions/demo_three-dimensions.py`` - 3D corner problem, Bank-Weiser
estimator, pure Dirichlet conditions, adaptive mesh refinement.
