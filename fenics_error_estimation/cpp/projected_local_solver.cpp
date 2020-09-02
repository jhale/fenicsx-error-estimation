// Copyright 2020, Jack S. Hale, Raphaël Bulle.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/mesh/cell_types.h>

namespace py = pybind11;

using namespace dolfinx;

template <typename T>
void projected_local_solver(
    const function::Function<T>& eta, const fem::Form<T>& a,
    const fem::Form<T>& L, const fem::Form<T>& L_eta,
    const fem::FiniteElement& element,
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
  Eigen::Matrix<T, Eigen::Dynamic, 1> be(element_space_dimension);
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_macro(2
                                              * element_space_dimension);

  // Prepare coefficients
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      a_coeffs = pack_coefficients(a);
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      L_coeffs = pack_coefficients(L);

  const std::vector<int> L_offsets = L.coefficients().offsets();
  Eigen::Array<T, Eigen::Dynamic, 1> L_coeff_array_macro(2 * L_offsets.back());

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in bilinear form a.");
  if (!L.all_constants_set())
    throw std::runtime_error("Unset constant in linear form L.");
  const Eigen::Array<T, Eigen::Dynamic, 1> a_constants
      = fem::pack_constants(a);
  const Eigen::Array<T, Eigen::Dynamic, 1> L_constants
      = fem::pack_constants(L);

  // Integrals
  const fem::FormIntegrals<T>& integrals = a.integrals();
  using type = fem::IntegralType;

  // Check assumptions on integrals.
  assert(a.integrals().num_integrals(type::cell) == 1);
  assert(a.integrals().num_integrals(type::interior_facet) == 0);
  assert(a.integrals().num_integrals(type::exterior_facet) == 0);
  assert(L.integrals().num_integrals(type::cell) == 1);
  assert(L.integrals().num_integrals(type::interior_facet) == 1);
  assert(L.integrals().num_integrals(type::exterior_facet) == 0);
  assert(L_eta.integrals().num_integrals(type::cell) == 1);
  assert(L_eta.integrals().num_integrals(type::interior_facet) == 0);
  assert(L_eta.integrals().num_integrals(type::exterior_facet) == 0);

  const auto a_kernel_domain_integral
      = a.integrals().get_tabulate_tensor(type::cell, 0);
  const auto L_kernel_domain_integral
      = L.integrals().get_tabulate_tensor(type::cell, 0);
  const auto L_kernel_interior_facet
      = L.integrals().get_tabulate_tensor(type::interior_facet, 0);
  const auto L_eta_kernel_domain_integral
      = L_eta.integrals().get_tabulate_tensor(type::cell, 0);

  // Prepare cell geometry
  const int gdim = mesh->geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs_macro(2 * num_dofs_g, gdim);

  // Iterate over active cells
  const int tdim = mesh->topology().dim();
  const auto map = mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local();

  // Needed for all integrals
  mesh->topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Needed for facet integrals
  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh->topology().get_facet_permutations();

  mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  mesh->topology_mutable().create_connectivity(tdim, tdim - 1);

  const auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  const auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  const auto cell_type = mesh->topology().cell_type();
  const int num_facets = mesh::cell_num_entities(cell_type, tdim - 1);

  for (int c = 0; c < num_cells; ++c)
  {
    // Get cell vertex coordinates
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

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
    for (int local_facet = 0; local_facet < num_facets; ++local_facet) {
      const std::int32_t f = c_f[local_facet];
      const auto f_c = f_to_c->links(f);
      assert(f_c.rows() < 3);

      if (f_c.rows() == 1)
      {
        // Is exterior facet
        const std::uint8_t perm = perms(local_facet, c);
        // TODO: Implement exterior facet term
        //L_kernel_exterior_facet(be.data(), L_coeff_array.data(),
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
          const auto* end = c_f.data() + c_f.rows();
          const auto* it = std::find(c_f.data(), end, f);
          assert(it != end);
          local_facets[k] = std::distance(c_f.data(), it);
        }

        // Orientation
        const std::array perm{perms(local_facets[0], f_c[0]),
                              perms(local_facets[1], f_c[1])};

        // Get cell geometry
        const auto x_dofs0 = x_dofmap.links(f_c[0]);
        const auto x_dofs1 = x_dofmap.links(f_c[1]);
        for (int k = 0; k < num_dofs_g; ++k)
        {
          for (int l = 0; l < gdim; ++l)
          {
            coordinate_dofs_macro(k, l) = x_g(x_dofs0[k], l);
            coordinate_dofs_macro(k + num_dofs_g, l) = x_g(x_dofs1[k], l);
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
          L_coeff_array_macro.segment(2 * L_offsets[i], num_entries)
              = L_coeff_cell0.segment(L_offsets[i], num_entries);
          L_coeff_array_macro.segment(L_offsets[i + 1] + L_offsets[i],
                                      num_entries)
              = L_coeff_cell1.segment(L_offsets[i], num_entries);
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
    // Is one of the current cell's facets on the boundary?
    


    // Perform projection and solve.

    // Project back.

    // Assemble into

  }
}

PYBIND11_MODULE(cpp, m)
{
  m.def("projected_local_solver", &projected_local_solver<PetscScalar>,
        "Local solves on projected finite element space");
}
