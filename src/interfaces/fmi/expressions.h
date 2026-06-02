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

#ifndef MOO_FMI_EXPRESSIONS_H
#define MOO_FMI_EXPRESSIONS_H

#include <base/util.h>
#include <base/export.h>

#include <functional>
#include <vector>
#include <stdint.h>

namespace FMI {

// Linear / Quadratic expressions
struct MOO_EXPORT ExprTerm {
    enum class Kind { Linear, Quadratic };

    Kind kind = Kind::Quadratic;
    uint32_t vref = 0;

    // tracking / offset function (evaluated at the collocation time)
    std::function<f64(f64)> reference = [](f64) { return 0.0; };

    // time-varying weight
    std::function<f64(f64)> weight = [](f64) { return 1.0; };

    static ExprTerm quadratic_term(uint32_t vref, f64 weight = 1.0);
    static ExprTerm tracking_term(uint32_t vref, std::function<f64(f64)> reference, f64 weight = 1.0);
    static ExprTerm linear_term(uint32_t vref, std::function<f64(f64)> reference = [](f64){ return 0.0; }, f64 weight = 1.0);
};

struct MOO_EXPORT Expr {
    std::vector<ExprTerm> terms;

    bool empty() const { return terms.empty(); }

    // evaluate at time t, given a callback that maps vref -> current FMU value
    f64 eval(f64 t, const std::function<f64(uint32_t vref)>& get_val) const;

    // partial derivative of the expression w.r.t. y_i (the vref's value)
    f64 deval_dy(const ExprTerm& term, f64 y_i, f64 t) const;
};

} // namespace FMI

#endif // MOO_FMI_EXPRESSIONS_H
