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

#ifndef MOO_FMI_PROBLEM_H
#define MOO_FMI_PROBLEM_H

#include <nlp/instances/gdop/problem.h>
#include <nlp/instances/gdop/gdop.h>

#include <interfaces/fmi/expressions.h>

namespace FMI {

// bounded output constraints
struct BoundedVRef {
    uint32_t vref;
    f64 lb = MINUS_INFINITY;
    f64 ub = PLUS_INFINITY;
};

// start values at initial time
struct FixedStartVRef {
    uint32_t vref;
    f64 value;
};

// nominal values
struct NominalVRef {
    uint32_t vref;
    f64 nominal;
};

// user-facing configuration / problem formulation
struct FMISettings {
    const char* path;
    const char* modelname;

    f64 t0 = 0.0;
    f64 tf = 1.0;
    int intervals = 15;
    int stage = 3;
    f64 tolerance = 1e-6;

    int l2bn_p1_it = 0;
    int l2bn_p2_it = 0;
    f64 l2bn_p2_lvl = 0.0;

    // objective terms (output vrefs)
    uint32_t* lagrange_vref = nullptr; // L(x, u, z, p, t)
    uint32_t* mayer_vref = nullptr; // M(xf, uf, zf, p, tf)

    // custom built expression for L
    Expr lagrange_expr;

    // p^L <= p <= p^U tunable parameters to include
    std::vector<BoundedVRef> parameter_vrefs;

    // u^L <= u <= u^U tunable controls to include
    std::vector<BoundedVRef> control_vrefs;

    // c^L <= c(x, u, z, p, t) <= c^U (appended to Lfg after residuals)
    std::vector<BoundedVRef> path_constraint_vrefs;

    // rf^L <= rf(xf, uf, zf, p, tf) <= rf^U (appended to Mrf after Mayer)
    std::vector<BoundedVRef> final_constraint_vrefs;

    // r0^L <= r0(x0, u0, z0, p, t0) <= r0^U (appended to Mrf after rf)
    std::vector<BoundedVRef> initial_constraint_vrefs;

    // (x, u, z)(0) fixed
    std::vector<FixedStartVRef> fixed_start_values;

    // nominals for each variable
    std::vector<NominalVRef> nominals;
};

class FMIData {
public:
    FMIData(FMISettings& settings);

    void print();
    void initialize(f64 t_start, f64 t_stop);
    std::vector<f64> get_initial_states();

    // pointwise evaluation / Jacobian (L?, f, g, c)
    void eval_point_lfg(const f64* xu, const f64* p, f64 time, f64* out);
    void jac_point_lfg (const f64* xu, const f64* p, f64 time, f64* out);

    // boundary evaluation / Jacobian (M?, rf) at final time
    void eval_point_mrf(const f64* xuf, const f64* p, f64 tf, f64* out);
    void jac_point_mrf (const f64* xuf, const f64* p, f64 tf, f64* out);

    // boundary evaluation / Jacobian (r0) at initial time
    void eval_point_r0(const f64* xu0, const f64* p, f64 t0, f64* out);
    void jac_point_r0 (const f64* xu0, const f64* p, f64 t0, f64* out);

    const std::vector<uint32_t>& get_xuz_vrefs() const;
    const std::vector<uint32_t>& get_p_vrefs() const;

    FMISettings& settings;
    std::unique_ptr<struct FMIData_priv> priv;
    std::vector<double> work;
};

class FullSweep : public GDOP::FullSweep {
public:
    FullSweep(GDOP::FullSweepLayout&& layout_in,
              const GDOP::ProblemConstants& pc,
              FMIData& fmi_data);

    void callback_eval(const f64* xu_nlp, const f64* p) override;
    void callback_jac (const f64* xu_nlp, const f64* p) override;
    void callback_hes (const f64* xu_nlp, const f64* p, const FixedField<f64, 2>& lagrange_factors, const f64* lambda) override;

    FMIData& fmi_data;
};

class BoundarySweep : public GDOP::BoundarySweep {
public:
    BoundarySweep(GDOP::BoundarySweepLayout&& layout_in,
                  const GDOP::ProblemConstants& pc,
                  FMIData& fmi_data);

    void callback_eval(const f64* xu0_nlp, const f64* xuf_nlp, const f64* p, f64 t0, f64 tf) override;
    void callback_jac (const f64* xu0_nlp, const f64* xuf_nlp, const f64* p, f64 t0, f64 tf) override;
    void callback_hes (const f64* xu0_nlp, const f64* xuf_nlp, const f64* p, f64 t0, f64 tf, const f64 mayer_factor, const f64* lambda) override;

    FMIData& fmi_data;
};

class Problem : public GDOP::Problem {
public:
    Problem(FMIData& fmi_data);
};

MOO_EXPORT void main_fmi(FMISettings& settings);

} // namespace FMI

#endif // MOO_FMI_PROBLEM_H
