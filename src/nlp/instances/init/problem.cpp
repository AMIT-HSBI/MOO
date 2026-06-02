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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include <base/log.h>

#include "problem.h"

namespace Init {

f64 value_or_zero(const FixedVector<f64>& values, int index)
{
    if (index < values.int_size()) {
        return values[index];
    }
    return 0.0;
}

f64 nominal_or_one(const FixedVector<f64>& nominals, int index)
{
    if (index < nominals.int_size() && std::isfinite(nominals[index]) && nominals[index] != 0.0) {
        return std::abs(nominals[index]);
    }
    return 1.0;
}

f64 p0_or_zero(const ProblemFormulation& formulation, int index)
{
    return value_or_zero(formulation.p0, index);
}

void Problem::eval_objective(const f64* y, const f64* p, f64& obj)
{
    switch (formulation.objective) {
        case Objective::ZERO:
            obj = 0.0;
            return;

        case Objective::LEAST_SQUARE_DEVIATION:
            obj = 0.0;
            for (int k = 0; k < formulation.p_size; k++) {
                const f64 nominal = nominal_or_one(formulation.p_nominal, k);
                const f64 dp = p[k] - p0_or_zero(formulation, k);
                obj += (dp / nominal) * (dp / nominal);
            }
            return;

        case Objective::USER:
            Log::error("Init::Problem objective is USER, but eval_objective was not overridden.");
            abort();
    }
}

void Problem::eval_grad_objective(const f64* y, const f64* p, f64* grad_y, f64* grad_p)
{
    std::fill(grad_y, grad_y + formulation.y_size, 0.0);
    std::fill(grad_p, grad_p + formulation.p_size, 0.0);

    switch (formulation.objective) {
        case Objective::ZERO:
            return;

        case Objective::LEAST_SQUARE_DEVIATION:
            for (int k = 0; k < formulation.p_size; k++) {
                const f64 nominal = nominal_or_one(formulation.p_nominal, k);
                grad_p[k] = 2.0 * (p[k] - p0_or_zero(formulation, k)) / (nominal * nominal);
            }
            return;

        case Objective::USER:
            Log::error("Init::Problem objective is USER, but eval_grad_objective was not overridden.");
            abort();
    }
}

void Problem::eval_hessian_objective(const f64* y, const f64* p, f64 obj_factor, f64* hes_values)
{
    switch (formulation.objective) {
        case Objective::ZERO:
        case Objective::LEAST_SQUARE_DEVIATION:
            return;

        case Objective::USER:
            Log::error("Init::Problem objective is USER, but eval_hessian_objective was not overridden.");
            abort();
    }
}

void Problem::eval_g(const f64* y, const f64* p, f64* g)
{
    if (formulation.g_size > 0) {
        Log::error("Init::Problem g_size is {}, but eval_g was not overridden.", formulation.g_size);
        abort();
    }
}

void Problem::eval_jacobian_g(const f64* y, const f64* p, f64* jac_g_values)
{
    if (formulation.g_size > 0) {
        Log::error("Init::Problem g_size is {}, but eval_jacobian_g was not overridden.", formulation.g_size);
        abort();
    }
}

void Problem::eval_hessian_constraints(const f64* y,
                                       const f64* p,
                                       const f64* lambda_f,
                                       const f64* lambda_g,
                                       f64* hes_values)
{

}

} // namespace Init
