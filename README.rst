#####################
Fast Helmholtz Solver
#####################

|License|_ |LastCommit|_ |CIStatus|_

.. |License| image:: https://img.shields.io/github/license/NaokiHori/FastHelmholtzSolver
.. _License: https://opensource.org/licenses/MIT

.. |LastCommit| image:: https://img.shields.io/github/last-commit/NaokiHori/FastHelmholtzSolver/main
.. _LastCommit: https://github.com/NaokiHori/FastHelmholtzSolver/commits/main

.. |CIStatus| image:: https://github.com/NaokiHori/FastHelmholtzSolver/actions/workflows/ci.yml/badge.svg?branch=main
.. _CIStatus: https://github.com/NaokiHori/FastHelmholtzSolver/actions/workflows/ci.yml

.. image:: https://github.com/NaokiHori/FastHelmholtzSolver/blob/main/thumbnail.png
   :width: 600

********
Overview
********

Massively parallelised spectral-based solver for Helmholtz, Poisson, and Laplace equations.

This library is intended to show a minimal example use of my other library `Simple Decomp <https://github.com/NaokiHori/SimpleDecomp>`_.

In particular, the API usage of domain decomposition, initialisation, pencil rotations, and parallel file IO are covered here.

**********
Dependency
**********

   * C compiler

   * MPI

   * FFTW3

***********
Quick start
***********

.. code-block:: console

   $ make all
   $ mpirun -n 128 ./a.out

This solves a Helmholtz equation on a cubic domain decomposed by 128^3 grid points with 128 processes.

The residual compared to the analytical solution is displayed, and the resulting three-dimensional scalar field is written to a ``NPY`` file.

