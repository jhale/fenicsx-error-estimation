// Copyright 2020, Jack S. Hale, Raphaël Bulle.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <algorithm>
#include <map>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>

#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/cell_types.h>

namespace py = pybind11;

using namespace dolfinx;

template <typename T, bool compute_error_solution = false>
void projected_local_solver(fem::Function<T>& eta_h,
                            const fem::Function<T>& e_D_h,
                            const fem::Form<T>& a, const fem::Form<T>& L,
                            const fem::Form<T>& L_eta,
                            const fem::FiniteElement& element,
                            const fem::ElementDofLayout& element_dof_layout,
                            const xt::pytensor<T, 2>& N,
                            const xt::pytensor<std::int32_t, 1>& entities,
                            fem::Function<T>& e_h)
{
  const auto mesh = a.mesh();
  assert(mesh == L.mesh());
  assert(mesh == L_eta.mesh());

  // Local tensors
  const int element_space_dimension = element.space_dimension();
  xt::xtensor<T, 2> Ae
      = xt::zeros<T>({element_space_dimension, element_space_dimension});
  xt::xtensor_fixed<T, xt::xshape<1>> etae = xt::zeros<double>({1});
  xt::xtensor<T, 1> be = xt::zeros<double>({element_space_dimension});
  xt::xtensor<T, 1> b_macro = xt::zeros<double>({2 * element_space_dimension});
  xt::xtensor<T, 2> Ae_0;
  xt::xtensor<T, 1> be_0, xe_0, xe;

  // Prepare coefficients
  const auto [a_coeffs, a_cstride] = pack_coefficients(a);
  const auto [L_coeffs, L_cstride] = pack_coefficients(L);

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
  assert(L.num_integrals(type::cell) == 1);
  assert(L_eta.num_integrals(type::interior_facet) == 0);
  assert(L_eta.num_integrals(type::exterior_facet) == 0);

  const auto& a_kernel_domain_integral = a.kernel(type::cell, -1);
  const auto& L_kernel_domain_integral = L.kernel(type::cell, -1);
  std::function<void(T*, const T*, const T*, const double*, const int*,
                     const std::uint8_t*)>
      L_kernel_exterior_facet;
  if (L.num_integrals(type::exterior_facet) == 1)
  {
    L_kernel_exterior_facet = L.kernel(type::exterior_facet, -1);
  }
  else
  {
    L_kernel_exterior_facet = nullptr;
  }
  const auto& L_kernel_interior_facet = L.kernel(type::interior_facet, -1);
  const auto& L_eta_kernel_domain_integral = L_eta.kernel(type::cell, -1);

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();
  std::vector<double> coordinate_dofs(num_dofs_g * 3);
  xt::xtensor<double, 3> coordinate_dofs_macro({2, num_dofs_g, 3});

  // dofmap and vector for inserting final error indicator
  const graph::AdjacencyList<std::int32_t>& dofmap_eta
      = eta_h.function_space()->dofmap()->list();
  std::shared_ptr<la::Vector<T>> eta_vec = eta_h.x();
  std::vector<T>& eta = eta_vec->mutable_array();

  // dofmap and vector of Dirichlet error
  const graph::AdjacencyList<std::int32_t>& dofmap_e_D
      = e_D_h.function_space()->dofmap()->list();
  std::shared_ptr<const la::Vector<T>> e_D_vec = e_D_h.x();
  const std::vector<T>& e_D = e_D_vec->array();

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
  // const std::vector<unsigned int>& cell_info
  //  = mesh->topology().get_cell_permutation_info();

  // Needed for facet integrals
  const std::vector<std::uint8_t>& perms
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

    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), i * 3));
    }
    std::fill(Ae.begin(), Ae.end(), 0.0);
    std::fill(be.begin(), be.end(), 0.0);

    const double* a_coeff_cell = a_coeffs.data() + cell * a_cstride;
    const double* L_coeff_cell = L_coeffs.data() + cell * L_cstride;
    a_kernel_domain_integral(Ae.begin(), a_coeff_cell, a_constants.data(),
                             coordinate_dofs.data(), nullptr, nullptr);
    L_kernel_domain_integral(be.begin(), L_coeff_cell, L_constants.data(),
                             coordinate_dofs.data(), nullptr, nullptr);

    // Loop over attached facets
    const auto c_f = c_to_f->links(c);
    bool cell_on_boundary = false;

    for (int local_facet = 0; local_facet < num_facets; ++local_facet)
    {
      const std::int32_t f = c_f[local_facet];
      const auto f_c = f_to_c->links(f);
      assert(f_c.size() < 3);

      if (f_c.size() == 1)
      {
        // Is exterior facet
        cell_on_boundary = true;
        if (L.num_integrals(type::exterior_facet) == 1)
        {
          const std::uint8_t perm = perms[c * c_f.size() + local_facet];
          // Exterior facet term
          L_kernel_exterior_facet(be.data(), L_coeff_cell, L_constants.data(),
                                  coordinate_dofs.data(), &local_facet, &perm);
        }
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
        const std::array perm{perms[f_c[0] * c_f.size() + local_facets[0]],
                              perms[f_c[1] * c_f.size() + local_facets[1]]};

        // Get cell geometry
        auto x_dofs0 = x_dofmap.links(f_c[0]);
        for (std::size_t i = 0; i < x_dofs0.size(); ++i)
        {
          std::copy_n(xt::view(x_g, x_dofs0[i]).begin(), 3,
                      xt::view(coordinate_dofs_macro, 0, i, xt::all()).begin());
        }
        auto x_dofs1 = x_dofmap.links(f_c[1]);
        for (std::size_t i = 0; i < x_dofs1.size(); ++i)
        {
          std::copy_n(xt::view(x_g, x_dofs1[i]).begin(), 3,
                      xt::view(coordinate_dofs_macro, 1, i, xt::all()).begin());
        }

        // Layout for the restricted coefficients is flattened
        // w[coefficient][restriction][dof]
        const double* L_coeff_cell0 = L_coeffs.data() + f_c[0] * L_cstride;
        const double* L_coeff_cell1 = L_coeffs.data() + f_c[1] * L_cstride;

        // Loop over coefficients for L
        for (std::size_t i = 0; i < L_offsets.size() - 1; ++i)
        {
          // Loop over entries for coefficient i
          const int num_entries = L_offsets[i + 1] - L_offsets[i];
          std::copy_n(L_coeff_cell0 + L_offsets[i], num_entries,
                      std::next(L_coeff_array_macro.begin(), 2 * L_offsets[i]));
          std::copy_n(L_coeff_cell1 + L_offsets[i], num_entries,
                      std::next(L_coeff_array_macro.begin(),
                                L_offsets[i + 1] + L_offsets[i]));
        }

        std::fill(b_macro.begin(), b_macro.end(), 0.0);

        L_kernel_interior_facet(
            b_macro.data(), L_coeff_array_macro.data(), L_constants.data(),
            coordinate_dofs_macro.data(), local_facets.data(), perm.data());

        // Assemble appropriate part of A_macro/b_macro into Ae/be.
        int local_cell = (f_c[0] == c ? 0 : 1);
        int offset = local_cell * element_space_dimension;
        be += xt::view(b_macro,
                       xt::range(offset, offset + element_space_dimension));
      }
    }

    // Get cell dofs Dirichlet error
    auto e_D_dofs = dofmap_e_D.links(c);

    // Apply boundary conditions.
    if (cell_on_boundary)
    {
      xt::xtensor<bool, 1> dofs_on_dirichlet_bc
          = xt::zeros<bool>({element_space_dimension});
      for (int local_facet = 0; local_facet < num_facets; ++local_facet)
      {
        const std::int32_t f = c_f[local_facet];
        if (std::binary_search(entities.begin(), entities.end(), f))
        {
          // Local facet is on Dirichlet boundary
          const std::vector<int>& local_dofs
              = element_dof_layout.entity_closure_dofs(tdim - 1, local_facet);
          for (std::size_t k = 0; k < local_dofs.size(); ++k)
          {
            dofs_on_dirichlet_bc[local_dofs[k]] = true;
          }
        }
      }

      for (int dof = 0; dof < element_space_dimension; ++dof)
      {
        if (dofs_on_dirichlet_bc[dof] == true)
        {
          xt::row(Ae, dof) = 0.0;
          xt::col(Ae, dof) = 0.0;
          Ae(dof, dof) = 1.0;
          be(dof) = e_D[e_D_dofs[dof]];
        }
      }
    }

    // Perform projection and solve.
    Ae_0 = xt::linalg::dot(xt::transpose(N), xt::linalg::dot(Ae, N));
    be_0 = xt::linalg::dot(xt::transpose(N), be);
    xe_0 = xt::linalg::solve(Ae_0, be_0);

    // Project back.
    xe = xt::linalg::dot(N, xe_0);

    // Compute indicator
    etae[0] = 0.0;
    L_eta_kernel_domain_integral(etae.data(), xe.data(), L_eta_constants.data(),
                                 coordinate_dofs.data(), nullptr, nullptr);

    // Assemble.
    const auto dofs_eta = dofmap_eta.links(c);
    eta[dofs_eta[0]] = etae(0);

    if constexpr (compute_error_solution)
    {
      const auto dofs_e = dofmap_e.links(c);
      for (std::size_t i = 0; i < dofs_e.size(); ++i)
      {
        e[dofs_e[i]] = xe[i];
      }
    }
  }
}

PYBIND11_MODULE(cpp, m)
{
  xt::import_numpy();
  m.def("projected_local_solver_error_solution",
        &projected_local_solver<PetscScalar, true>,
        "Local solves on projected finite element space. Computes Bank-Weiser "
        "error solution.")
      .def("projected_local_solver_no_error_solution",
           &projected_local_solver<PetscScalar, false>,
           "Local solves on projected finite element space. Does not compute "
           "Bank-Weiser error solution.");
}
