// Copyright 2015 - 2018, Jack S. Hale.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/LocalAssembler.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>

namespace py = pybind11;

using namespace dolfin;

void projected_local_solver(dolfin::Function& e,
                            const dolfin::Form& a,
                            const dolfin::Form& L,
                            const py::EigenDRef<const Eigen::MatrixXd> N,
                            const std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
    // TODO: More dolfin_assert.
    // Boiler plate setup.
    std::shared_ptr<const dolfin::Mesh> mesh = a.mesh();
    UFC a_ufc(a);
    UFC L_ufc(L);

    // TODO: Check size of N wrt to compatibility with a and L.
    typedef std::vector<std::shared_ptr<const GenericFunction>> coefficient_t;
    const coefficient_t coefficients_a = a.coefficients();
    const coefficient_t coefficients_L = L.coefficients();

    // Local element tensors, solver.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::RowMajor> A_e, A_e_0, b_e, b_e_0, x_e, x_e_0, NT;
    Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>> solver;

    // To store global error result.
    //auto e = std::make_shared<Function>(L.function_space(0));
    auto e_vector = e.vector();

    // More boiler plate.
    // TODO: Check working with bilinear forms leading to square matrices?
    std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps_a
        = {{a.function_space(0)->dofmap(), a.function_space(1)->dofmap()}};
    std::shared_ptr<const GenericDofMap> dofmap_L = L.function_space(0)->dofmap();

    std::vector<double> coordinate_dofs;
    ufc::cell ufc_cell;
    std::vector<ArrayView<const dolfin::la_index>> dofs(1); 

    // Dirichlet Boundary Conditions
    // Clearly only works with square matrices.
    std::vector<DirichletBC::Map> boundary_values(1);
    for (auto bc : bcs) {
        bc->get_boundary_values(boundary_values[0]);
        if (MPI::size(mesh->mpi_comm()) > 1 && bc->method() != "pointwise")
            bc->gather(boundary_values[0]);
    }

    // Used in Dirichlet BC application.
    auto dofs_a0 = dofmaps_a[0]->cell_dofs(0);
    auto dofs_a1 = dofmaps_a[1]->cell_dofs(0);
    // Used in insertion into global vector.
    auto dofs_L = dofmap_L->cell_dofs(0);

    A_e.resize(dofs_a0.size(), dofs_a1.size());
    b_e.resize(dofs_L.size(), 1);

    NT = N.transpose().eval();

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        dolfin_assert(!cell->is_ghost());

        // Used in Dirichlet BC application.
        auto dofs_a0 = dofmaps_a[0]->cell_dofs(cell->index());
        auto dofs_a1 = dofmaps_a[1]->cell_dofs(cell->index());
        // Used in insertion into global vector.
        auto dofs_L = dofmap_L->cell_dofs(cell->index());
        dofs[0] = ArrayView<const dolfin::la_index>(dofs_L.size(), dofs_L.data());

        cell->get_coordinate_dofs(coordinate_dofs);

        LocalAssembler::assemble(A_e, a_ufc, coordinate_dofs, ufc_cell, *cell,
                                 a.cell_domains().get(),
                                 a.exterior_facet_domains().get(),
                                 a.interior_facet_domains().get());

        LocalAssembler::assemble(b_e, L_ufc, coordinate_dofs, ufc_cell, *cell,
                                 L.cell_domains().get(),
                                 L.exterior_facet_domains().get(),
                                 L.interior_facet_domains().get());

        // Apply boundary conditions (taken from SystemAssembler).
        for (unsigned int i = 0; i < A_e.cols(); ++i) {
            const dolfin::la_index ii = dofs_a0[i];
            // NOTE: Implicit assumption dofs_a1[i] == dofs_a0[i].
            DirichletBC::Map::const_iterator bc_value = boundary_values[0].find(ii);
            if (bc_value != boundary_values[0].end())
            {
                // Zero row.
                A_e.row(i).setZero();
                // Modify RHS.
                b_e -= A_e.col(i)*bc_value->second;
                // Zero column.
                A_e.col(i).setZero();
                // Place 1 on diagonal.
                A_e(i, i) = 1.0;
                // BC on RHS.
                b_e(i) = bc_value->second;
            }
        }

        // Apply projection to V_0.
        A_e_0.noalias() = NT*A_e*N;
        b_e_0.noalias() = NT*b_e;

        // Solve
        solver.compute(A_e_0);
        x_e_0 = solver.solve(b_e_0);

        // Apply projection back to V_f.
        x_e.noalias() = N*x_e_0;

        // Insert into vector of Function.
        e_vector->add_local(x_e.data(), dofs); 
    }
    e_vector->apply("insert");
}

PYBIND11_MODULE(cpp, m)
{
    m.def("projected_local_solver", &projected_local_solver, "Local solves on projected finite element space");
}
