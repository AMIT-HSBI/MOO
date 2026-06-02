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

#include <interfaces/fmi/problem.h>

#include <cmath>

int main() {
    FMI::FMISettings settings;
    settings.path = MULTILINK_FMU_PATH;
    settings.modelname = "Multilink";
    settings.t0 = 0.0;
    settings.tf = 2.0;
    settings.intervals = 400;
    settings.stage = 1;
    settings.tolerance = 1e-6;

    // control
    uint32_t vref_u = 620756992;

    // states
    uint32_t vref_phi = 33554432;
    uint32_t vref_w_a = 33554433;

    f64 factor_u = 1e-12;
    f64 factor_phi = 1e3;
    f64 phi_0 = -0.286848;
    f64 phi_A = 0.13;
    f64 phi_f = 1.0;

    settings.control_vrefs.push_back( { vref_u, -2e6, 2e6 } );
    settings.fixed_start_values = { {vref_phi, phi_0}, {vref_w_a, 2 * M_PI * phi_A * phi_f} };
    settings.nominals = { { vref_u, 2e6 } };

    auto u_penalty = FMI::ExprTerm::quadratic_term(vref_u, factor_u);
    auto phi_penalty = FMI::ExprTerm::tracking_term(vref_phi, [&](f64 t) -> f64 { return phi_0 + phi_A * sin(2 * M_PI * t * phi_f);}, factor_phi);

    settings.lagrange_expr.terms = { phi_penalty, u_penalty };

    FMI::main_fmi(settings);

    return EXIT_SUCCESS;
}
