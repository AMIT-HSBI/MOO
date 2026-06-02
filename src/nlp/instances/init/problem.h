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

#ifndef MOO_INIT_PROBLEM_H
#define MOO_INIT_PROBLEM_H

#include <base/export.h>
#include <base/fixed_vector.h>
#include <base/nlp_structs.h>

namespace Init {

enum class Objective {
    ZERO,
    LEAST_SQUARE_DEVIATION,
    USER
};

struct MOO_EXPORT ProblemFormulation {
    int y_size = 0;
    int p_size = 0;
    int f_size = 0;
    int g_size = 0;

    Objective objective = Objective::ZERO;

    FixedVector<f64> y0;
    FixedVector<f64> p0;
    FixedVector<f64> dp0;

    FixedVector<Bounds> y_bounds;
    FixedVector<Bounds> p_bounds;
    FixedVector<Bounds> dp_bounds;
    FixedVector<Bounds> g_bounds;

    f64 obj_nominal = 1.0;
    FixedVector<f64> y_nominal;
    FixedVector<f64> p_nominal;
    FixedVector<f64> dp_nominal;
    FixedVector<f64> f_nominal;
    FixedVector<f64> g_nominal;

    FixedVector<int> jac_f_rows;
    FixedVector<int> jac_f_cols;
    FixedVector<int> jac_g_rows;
    FixedVector<int> jac_g_cols;
    FixedVector<int> hes_rows;
    FixedVector<int> hes_cols;
};

struct MOO_EXPORT Result {
    f64 objective = 0.0;
    f64 f_l2_norm = 0.0;
    f64 f_max_error = 0.0;
    f64 g_max_violation = 0.0;
    FixedVector<f64> y;
    FixedVector<f64> dp;
    FixedVector<f64> p;
    FixedVector<f64> constraints;
    FixedVector<f64> lambda;
    FixedVector<f64> z_lb;
    FixedVector<f64> z_ub;
};

class MOO_EXPORT Problem {
public:
    ProblemFormulation formulation;

    virtual ~Problem() = default;

    virtual void eval_objective(const f64* y, const f64* p, f64& obj);
    virtual void eval_grad_objective(const f64* y, const f64* p, f64* grad_y, f64* grad_p);
    virtual void eval_hessian_objective(const f64* y, const f64* p, f64 obj_factor, f64* hes_values);

    virtual void eval_f(const f64* y, const f64* p, f64* f) = 0;
    virtual void eval_g(const f64* y, const f64* p, f64* g);

    virtual void eval_jacobian_f(const f64* y, const f64* p, f64* jac_f_values) = 0;
    virtual void eval_jacobian_g(const f64* y, const f64* p, f64* jac_g_values);

    virtual void eval_hessian_constraints(const f64* y,
                                          const f64* p,
                                          const f64* lambda_f,
                                          const f64* lambda_g,
                                          f64* hes_values);
};

} // namespace Init

#endif // MOO_INIT_PROBLEM_H
