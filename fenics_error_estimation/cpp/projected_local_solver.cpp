// Copyright 2020, Jack S. Hale, Raphaël Bulle.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <algorithm>
#include <map>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/cell_types.h>

namespace py = pybind11;

using namespace dolfinx;

template <typename T>
void projected_local_solver(
    fem::Function<T>& eta_h, fem::Function<T>& e_h, const fem::Form<T>& a, const fem::Form<T>& L,
    const fem::Form<T>& L_eta, const fem::FiniteElement& element,
    const fem::ElementDofLayout& element_dof_layout,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>
        N,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
        entities)
{
  const auto mesh = a.mesh();
  assert(mesh == L.mesh());
  assert(mesh == L_eta.mesh());

  // These assertions fail because these forms have empty _function_spaces
  // assert(a.rank() == 2);
  // assert(L.rank() == 1);
  // assert(L_eta.rank() == 1);

  // Local tensors
  const int element_space_dimension = element.space_dimension();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae(
      element_space_dimension, element_space_dimension);
  Eigen::Matrix<T, 1, 1> etae;
  Eigen::Matrix<T, Eigen::Dynamic, 1> be(element_space_dimension);
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_macro(2 * element_space_dimension);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae_0, be_0,
      xe_0, xe;
  Eigen::FullPivLU<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      solver;

  // Prepare coefficients
  const dolfinx::array2d<double>
      a_coeffs = pack_coefficients(a);
  const dolfinx::array2d<double>
      L_coeffs = pack_coefficients(L);

  const std::vector<int> L_offsets = L.coefficient_offsets();
  std::vector<T> L_coeff_array_macro(2 * L_offsets.back());

  // Prepare constants
  const std::vector<double> a_constants = fem::pack_constants(a);
  const std::vector<double> L_constants = fem::pack_constants(L);
  const std::vector<double> L_eta_constants = fem::pack_constants(L_eta);

  // Check assumptions on integrals.
  using type = fem::IntegralType;
  assert(a.num_integrals(type::cell) == 1);
  assert(a.num_integrals(type::interior_facet) == 0);
  assert(a.num_integrals(type::exterior_facet) == 0);
  assert(L.num_integrals(type::cell) == 1);
  assert(L.num_integrals(type::interior_facet) == 1);
  assert(L.num_integrals(type::exterior_facet) == 0);
  assert(L.num_integrals(type::cell) == 1);
  assert(L_eta.num_integrals(type::interior_facet) == 0);
  assert(L_eta.num_integrals(type::exterior_facet) == 0);

  const auto& a_kernel_domain_integral = a.kernel(type::cell, -1);
  const auto& L_kernel_domain_integral = L.kernel(type::cell, -1);
  const auto& L_kernel_interior_facet = L.kernel(type::interior_facet, -1);
  const auto& L_eta_kernel_domain_integral = L_eta.kernel(type::cell, -1);

  // Prepare cell geometry
  const int gdim = mesh->geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const dolfinx::array2d<double>& x_g
      = mesh->geometry().x();
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);
  std::vector<double> coordinate_dofs_macro(2 * num_dofs_g * gdim);

  // dofmap and vector for inserting final error indicator
  const graph::AdjacencyList<std::int32_t>& dofmap_eta
      = eta_h.function_space()->dofmap()->list();
  std::shared_ptr<la::Vector<T>> eta_vec = eta_h.x();
  std::vector<T>& eta = eta_vec->mutable_array();

  // dofmap and vector for inserting error solution
  const graph::AdjacencyList<std::int32_t>& dofmap_e
      = e_h.function_space()->dofmap()->list();
  std::shared_ptr<la::Vector<T>> e_vec = e_h.x();
  std::vector<T>& e = e_vec->mutable_array();
  
  // Iterate over active cells
  const int tdim = mesh->topology().dim();
  const auto map = mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local();

  // Needed for all integrals
  mesh->topology_mutable().create_entity_permutations();
  const std::vector<unsigned int>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Needed for facet integrals
  const std::vector<std::uint8_t>& perms
      = mesh->topology().get_facet_permutations();

  mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  mesh->topology_mutable().create_connectivity(tdim, tdim - 1);

  const auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  const auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  const int offset_g = gdim * num_dofs_g;

  const auto cell_type = mesh->topology().cell_type();
  const int num_facets = mesh::cell_num_entities(cell_type, tdim - 1);

  // Hacky map for CG -> DG dof layout being different.
  // Only works for CG2/DG2 on triangles.
  // Once entity closure dofs is coded this should be unnecessary.
  std::map<int, int> mapping{{3, 4}, {4, 3}, {5, 1}};

  for (int c = 0; c < num_cells; ++c)
  {
    // Get cell vertex coordinates
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
    {
      std::copy_n(x_g.row(x_dofs[i]).data(), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }
    Ae.setZero();
    be.setZero();

    const auto a_coeff_array = a_coeffs.row(c);
    const auto L_coeff_array = L_coeffs.row(c);
    a_kernel_domain_integral(Ae.data(), a_coeff_array.data(),
                             a_constants.data(), coordinate_dofs.data(),
                             nullptr, nullptr, cell_info[c]);
    L_kernel_domain_integral(be.data(), L_coeff_array.data(),
                             L_constants.data(), coordinate_dofs.data(),
                             nullptr, nullptr, cell_info[c]);

    // Loop over attached facets
    const auto c_f = c_to_f->links(c);
    for (int local_facet = 0; local_facet < num_facets; ++local_facet)
    {
      const std::int32_t f = c_f[local_facet];
      const auto f_c = f_to_c->links(f);
      assert(f_c.size() < 3);

      if (f_c.size() == 1)
      {
        // Is exterior facet
        // const std::uint8_t perm = perms(local_facet, c);
        // TODO: Implement exterior facet term
        // L_kernel_exterior_facet(be.data(), L_coeff_array.data(),
        //                        L_constants.data(), coordinate_dofs.data(),
        //                        &local_facet, &perm, cell_info[c]);
      }
      else
      {
        // Is interior facet
        std::array<int, 2> local_facets;
        for (int k = 0; k < 2; ++k)
        {
          const auto c_f = c_to_f->links(f_c[k]);
          const auto* end = c_f.data() + c_f.size();
          const auto* it = std::find(c_f.data(), end, f);
          assert(it != end);
          local_facets[k] = std::distance(c_f.data(), it);
        }

        // Orientation
        const std::array perm{perms[local_facets[0], f_c[0]],
                              perms[local_facets[1], f_c[1]]};

        // Get cell geometry
        const auto x_dofs0 = x_dofmap.links(f_c[0]);
        const auto x_dofs1 = x_dofmap.links(f_c[1]);
        for (int k = 0; k < num_dofs_g; ++k)
        {
          for (int l = 0; l < gdim; ++l)
          {
            coordinate_dofs_macro[k * gdim + l] = x_g(x_dofs0[k], l);
            coordinate_dofs_macro[offset_g + k * gdim + l] = x_g(x_dofs1[k], l);
          }
        }

        // Layout for the restricted coefficients is flattened
        // w[coefficient][restriction][dof]
        const auto L_coeff_cell0 = L_coeffs.row(f_c[0]);
        const auto L_coeff_cell1 = L_coeffs.row(f_c[1]);

        // Loop over coefficients for L
        for (std::size_t i = 0; i < L_offsets.size() - 1; ++i)
        {
          // Loop over entries for coefficient i
          const int num_entries = L_offsets[i + 1] - L_offsets[i];
          std::copy_n(L_coeff_cell0.data() + L_offsets[i], num_entries,
                      std::next(L_coeff_array_macro.begin(), 2 * L_offsets[i]));
          std::copy_n(L_coeff_cell1.data() + L_offsets[i], num_entries,
                      std::next(L_coeff_array_macro.begin(),
                                L_offsets[i + 1] + L_offsets[i]));
        }

        b_macro.setZero();

        L_kernel_interior_facet(
            b_macro.data(), L_coeff_array_macro.data(), L_constants.data(),
            coordinate_dofs_macro.data(), local_facets.data(), perm.data(),
            cell_info[f_c[0]]);

        // Assemble appropriate part of A_macro/b_macro into Ae/be.
        int local_cell = (f_c[0] == c ? 0 : 1);
        int offset = local_cell * element_space_dimension;
        be += b_macro.block(offset, 0, be.rows(), 1);
      }
    }

    // Apply boundary conditions.
    Eigen::Array<bool, Eigen::Dynamic, 1> dofs_on_dirichlet_bc(
        element_space_dimension);
    dofs_on_dirichlet_bc.setZero();
    for (int local_facet = 0; local_facet < num_facets; ++local_facet)
    {
      const std::int32_t f = c_f[local_facet];
      const auto* end = entities.data() + entities.size();
      const auto* it = std::find(entities.data(), end, f);
      if (it != end)
      {
        // Local facet is on Dirichlet boundary
        const std::vector<int> local_dofs
            = element_dof_layout.entity_closure_dofs(tdim - 1, local_facet);
        for (std::size_t k = 0; k < local_dofs.size(); ++k)
        {
          dofs_on_dirichlet_bc[mapping[local_dofs[k]]] = true;
        }
      }
    }

    for (int dof = 0; dof < element_space_dimension; ++dof)
    {
      if (dofs_on_dirichlet_bc[dof] == true)
      {
        Ae.row(dof).setZero();
        Ae.col(dof).setZero();
        Ae(dof, dof) = 1.0;
        be(dof) = 0.0;
      }
    }

    // Perform projection and solve.
    Ae_0 = N.transpose() * Ae * N;
    be_0 = N.transpose() * be;
    solver.compute(Ae_0);
    xe_0 = solver.solve(be_0);

    // Project back.
    xe = N * xe_0;

    // Compute indicator
    etae.setZero();
    L_eta_kernel_domain_integral(etae.data(), xe.data(), L_eta_constants.data(),
                                 coordinate_dofs.data(), nullptr, nullptr,
                                 cell_info[c]);

    // Assemble.
    const auto dofs_eta = dofmap_eta.links(c);
    eta[dofs_eta] = etae(0);

    const auto dofs_e = dofmap_e.links(c);
    for (int i = 0; i < dofs_e.size(); ++i) {
      e[dofs_e[i]] = xe(i);
    } 
  }
}

PYBIND11_MODULE(cpp, m)
{
  m.def("projected_local_solver", &projected_local_solver<PetscScalar>,
        "Local solves on projected finite element space");
}
