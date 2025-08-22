// SPDX-License-Identifier: LGPL-3.0-or-later
//
// This file is part of MOO - Modelica / Model Optimizer
// Copyright (C) 2025 University of Applied Sciences and Arts
// Bielefeld, Faculty of Engineering and Mathematics
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include "radau_integrator.h"

namespace Simulation {

static void* radau_integrator = nullptr;

RadauIntegrator::RadauIntegrator(/* generic Integrator */
                                 ODEFunction ode_fn,
                                 std::vector<f64> dense_output_grid,
                                 f64* x_start_values,
                                 int x_size,
                                 void* user_data,
                                 f64* parameters,
                                 int p_size,
                                 ControlTrajectory* controls,
                                 JacobianFunction jac_fn,
                                 JacobianFormat jfmt,
                                 int* row,
                                 int* col,
                                 int nnz,
                                 /* Radau specific */
                                 RadauScheme scheme,
                                 f64 h_init,
                                 f64 atol,
                                 f64 rtol)
    : Integrator(ode_fn, dense_output_grid, x_start_values, x_size, user_data,
                 parameters, p_size, controls, jac_fn, jfmt, row, col, nnz),
      scheme(scheme),
      h_init(h_init),
      atol(atol),
      rtol(rtol),
      return_code(0),
      dense_output_index(0) {}

extern "C" void radau_fcn_wrapper(
    int* n,
    f64* t,
    f64* y,
    f64* dydx)
{
    auto* self = static_cast<Simulation::Integrator*>(radau_integrator);
    self->get_ode(*t, y, dydx);
}

extern "C" void radau_dense_jac_wrapper(
    int* n,
    f64* t,
    f64* x,
    int* ml,
    int* mu,
    f64* pd,
    f64* pdata)
{
    auto* self = static_cast<Simulation::Integrator*>(radau_integrator);
    self->get_dense_jacobian(*t, x, pd);
}

extern "C" void no_mass(int* n, int* m, f64* data) {}

extern "C" void dense_output(
    int* nr,
    f64* told,
    f64* t,
    f64* x,
    f64* cont,
    int* lrc,
    int* n,
    f64* rpar,
    int* ipar,
    int* irtrn)
{
    auto* integrator = static_cast<Simulation::RadauIntegrator*>(radau_integrator);
    if (!integrator) return;

    const std::vector<f64>& grid = integrator->dense_output_grid;
    size_t& idx = integrator->dense_output_index;
    auto& output = integrator->output;
    auto u_t_eval = integrator->u.raw(); // set by set_controls_only to u(t_eval)

    while (idx < grid.size()) {
        f64 t_eval = grid[idx];
        if (t_eval > *t) break;
        if (t_eval < *told) {
            idx++;
            continue;
        }

        output->t.push_back(t_eval);

        for (int x_idx = 0; x_idx < integrator->x_size; x_idx++) {
            int fortran_x = x_idx + 1;
            output->x[x_idx].push_back(contra_(&fortran_x, &t_eval, cont, lrc));
        }

        integrator->set_controls(t_eval);

        for (int u_idx = 0; u_idx < integrator->u_size; ++u_idx) {
            output->u[u_idx].push_back(u_t_eval[u_idx]);
        }

        idx++;
    }

    *irtrn = 0;
}

int RadauIntegrator::internal_simulate() {
    f64 t0 = dense_output_grid[0];
    f64 tf = dense_output_grid.back();

    int ijac = (jac_func) ? 1 : 0;
    
    int lwork = 100 + 20 * x_size;
    int liwork = 100 + 20 * x_size;

    std::vector<f64> work(lwork, 0.0);
    std::vector<int> iwork(liwork, 0);

    switch (scheme)
    {
        case RadauScheme::ONE:
            iwork[10] = 1; /* min_m */
            iwork[11] = 1; /* max_m */
            iwork[12] = 1; /* start_m */
            break;
        
        case RadauScheme::FIVE:
            iwork[10] = 3; /* min_m */
            iwork[11] = 3; /* max_m */
            iwork[12] = 3; /* start_m */
            break;

        case RadauScheme::NINE:
            iwork[10] = 5; /* min_m */
            iwork[11] = 5; /* max_m */
            iwork[12] = 5; /* start_m */
            break;

        case RadauScheme::THIRTEEN:
            iwork[10] = 7; /* min_m */
            iwork[11] = 7; /* max_m */
            iwork[12] = 7; /* start_m */
            break;

        case RadauScheme::ADAPTIVE:
        default:
            iwork[10] = 1; /* min_m */
            iwork[11] = 7; /* max_m */
            iwork[12] = 5; /* start_m */
            break;
    }

    radau_integrator = this;

    radau_(
        &x_size,
        radau_fcn_wrapper,
        &t0,
        x_start_values,
        &tf,
        &h_init,
        &rtol,
        &atol,
        &itol,
        radau_dense_jac_wrapper,
        &ijac,
        &mljac,
        &mujac,
        no_mass,
        &imas,
        &mlmas,
        &mumas,
        dense_output,
        &iout,
        work.data(),
        &lwork,
        iwork.data(),
        &liwork,
        rpar,
        ipar,
        &return_code
    );

    radau_integrator = nullptr;

    if (return_code < 0) {
        return return_code;
    }

    return 0;
}

} // namespace Simulation
