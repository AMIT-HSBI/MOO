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

#include <base/util.h>
#include <interfaces/fmi/expressions.h>

#include <stdint.h>

namespace FMI {

// out = weight * x[vref]**2
ExprTerm ExprTerm::quadratic_term(uint32_t vref, f64 weight)
{
    ExprTerm t;
    t.kind = ExprTerm::Kind::Quadratic;
    t.vref = vref;
    t.weight = [weight](f64) { return weight; };
    t.reference = [](f64) { return 0.0; };
    return t;
}

// out = weight * (x[vref] - ref(t))**2
ExprTerm ExprTerm::tracking_term(uint32_t vref, std::function<f64(f64)> reference, f64 weight)
{
    ExprTerm t;
    t.kind = ExprTerm::Kind::Quadratic;
    t.vref = vref;
    t.weight = [weight](f64) { return weight; };
    t.reference = std::move(reference);
    return t;
}

// out = weight * (x[vref] - ref(t))
ExprTerm ExprTerm::linear_term(uint32_t vref, std::function<f64(f64)> reference, f64 weight)
{
    ExprTerm t;
    t.kind = ExprTerm::Kind::Linear;
    t.vref = vref;
    t.weight = [weight](f64) { return weight; };
    t.reference = std::move(reference);
    return t;
}

// evaluate at time t, given a callback that maps vref -> current FMU value
f64 Expr::eval(f64 t, const std::function<f64(uint32_t vref)>& get_val) const
{
    f64 result = 0.0;
    for (const auto& term : terms) {
        f64 y = get_val(term.vref);
        f64 ref = term.reference(t);
        f64 w = term.weight(t);
        f64 r = y - ref;
        result += (term.kind == ExprTerm::Kind::Quadratic) ? w * r * r : w * r;
    }
    return result;
}

// partial derivative of the expression w.r.t. y_i (the vref's value)
f64 Expr::deval_dy(const ExprTerm& term, f64 y_i, f64 t) const
{
    f64 w = term.weight(t);
    f64 ref = term.reference(t);
    return (term.kind == ExprTerm::Kind::Quadratic) ? w * 2.0 * (y_i - ref) : w;
}

} // namespace FMI
