// Copyright 2015 - 2018, Jack S. Hale.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <dolfinx/fem/Form.h>

namespace py = pybind11;

using namespace dolfinx;

template <typename T>
void projected_local_solver(const fem::Form<T>& a,
                            const fem::Form<T>& L,
                            const py::EigenDRef<const Eigen::MatrixXd> N)
{
}

PYBIND11_MODULE(cpp, m)
{
    m.def("projected_local_solver", &projected_local_solver<PetscScalar>, "Local solves on projected finite element space");
}
