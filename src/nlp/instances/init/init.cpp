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
#include <cmath>
#include <cstdlib>

#include <base/log.h>
#include <nlp/nlp_scaling.h>

#include "init.h"

namespace Init {

namespace {

Bounds bounds_or_unbounded(const FixedVector<Bounds>& bounds, int index)
{
    if (index < bounds.int_size()) {
        return bounds[index];
    }
    return Bounds{};
}

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

bool valid_optional_size(const char* name, int size, int expected)
{
    if (size == 0 || size == expected) {
        return true;
    }

    Log::error("Init::ProblemFormulation {} has size {}, expected 0 or {}.", name, size, expected);
    return false;
}

bool has_nominal_scaling(const ProblemFormulation& formulation)
{
    return !formulation.y_nominal.empty()
        || !formulation.p_nominal.empty()
        || !formulation.dp_nominal.empty()
        || !formulation.f_nominal.empty()
        || !formulation.g_nominal.empty();
}

f64 objective_nominal_or_one(f64 obj_nominal)
{
    if (std::isfinite(obj_nominal) && obj_nominal != 0.0) {
        return std::abs(obj_nominal);
    }
    return 1.0;
}

} // namespace

Init::Init(Problem& problem)
    : problem(problem),
      p_buffer(FixedVector<f64>(problem.formulation.p_size)),
      grad_y_buffer(FixedVector<f64>(problem.formulation.y_size)),
      grad_p_buffer(FixedVector<f64>(problem.formulation.p_size)),
      jac_f_buffer(FixedVector<f64>(problem.formulation.jac_f_rows.size())),
      jac_g_buffer(FixedVector<f64>(problem.formulation.jac_g_rows.size()))
{
    validate_formulation();
    initialize_objective_hessian_structure();
}

const Result& Init::get_result() const
{
    return result;
}

std::shared_ptr<::NLP::Scaling> Init::get_scaling()
{
    const auto& form = formulation();

    if (!has_nominal_scaling(form)) {
        return std::make_shared<::NLP::NoScaling>();
    }

    FixedVector<f64> x_nominal(number_vars_internal());
    FixedVector<f64> g_nominal(number_constraints_internal());

    for (int y = 0; y < form.y_size; y++) {
        x_nominal[y] = nominal_or_one(form.y_nominal, y);
    }

    for (int p = 0; p < form.p_size; p++) {
        x_nominal[form.y_size + p] = form.dp_nominal.empty()
                                   ? nominal_or_one(form.p_nominal, p)
                                   : nominal_or_one(form.dp_nominal, p);
    }

    for (int f = 0; f < form.f_size; f++) {
        g_nominal[f] = nominal_or_one(form.f_nominal, f);
    }

    for (int g = 0; g < form.g_size; g++) {
        g_nominal[form.f_size + g] = nominal_or_one(form.g_nominal, g);
    }

    return std::make_shared<::NLP::NominalScaling>(
        std::move(x_nominal),
        std::move(g_nominal),
        objective_nominal_or_one(form.obj_nominal));
}

void Init::get_sizes(int& number_vars, int& number_constraints)
{
    number_vars = number_vars_internal();
    number_constraints = number_constraints_internal();
}

void Init::get_bounds(FixedVector<f64>& x_lb,
                      FixedVector<f64>& x_ub,
                      FixedVector<f64>& g_lb,
                      FixedVector<f64>& g_ub)
{
    const auto& form = formulation();

    for (int y = 0; y < form.y_size; y++) {
        const Bounds bounds = bounds_or_unbounded(form.y_bounds, y);
        x_lb[y] = bounds.lb;
        x_ub[y] = bounds.ub;
    }

    for (int p = 0; p < form.p_size; p++) {
        const Bounds dp_bounds = bounds_or_unbounded(form.dp_bounds, p);
        const Bounds p_bounds = bounds_or_unbounded(form.p_bounds, p);
        const f64 p0 = p0_or_zero(form, p);

        x_lb[form.y_size + p] = std::max(dp_bounds.lb, p_bounds.lb - p0);
        x_ub[form.y_size + p] = std::min(dp_bounds.ub, p_bounds.ub - p0);
    }

    for (int f = 0; f < form.f_size; f++) {
        g_lb[f] = 0.0;
        g_ub[f] = 0.0;
    }

    for (int g = 0; g < form.g_size; g++) {
        const Bounds bounds = bounds_or_unbounded(form.g_bounds, g);
        g_lb[form.f_size + g] = bounds.lb;
        g_ub[form.f_size + g] = bounds.ub;
    }
}

void Init::get_initial_guess(bool init_x,
                             FixedVector<f64>& x_init,
                             bool init_lambda,
                             FixedVector<f64>& lambda_init,
                             bool init_z,
                             FixedVector<f64>& z_lb_init,
                             FixedVector<f64>& z_ub_init)
{
    const auto& form = formulation();

    if (init_x) {
        for (int y = 0; y < form.y_size; y++) {
            x_init[y] = value_or_zero(form.y0, y);
        }

        for (int p = 0; p < form.p_size; p++) {
            x_init[form.y_size + p] = value_or_zero(form.dp0, p);
        }
    }

    if (init_lambda) {
        lambda_init.fill_zero();
    }

    if (init_z) {
        z_lb_init.fill_zero();
        z_ub_init.fill_zero();
    }
}

void Init::get_nnz(int& nnz_jac, int& nnz_hes)
{
    const auto& form = formulation();
    nnz_jac = form.jac_f_rows.int_size() + form.jac_g_rows.int_size();
    nnz_hes = form.hes_rows.int_size();
}

void Init::get_jac_sparsity(FixedVector<int>& i_row_jac, FixedVector<int>& j_col_jac)
{
    const auto& form = formulation();
    int offset = 0;

    for (int nz = 0; nz < form.jac_f_rows.int_size(); nz++) {
        i_row_jac[offset] = form.jac_f_rows[nz];
        j_col_jac[offset] = form.jac_f_cols[nz];
        offset++;
    }

    for (int nz = 0; nz < form.jac_g_rows.int_size(); nz++) {
        i_row_jac[offset] = form.f_size + form.jac_g_rows[nz];
        j_col_jac[offset] = form.jac_g_cols[nz];
        offset++;
    }
}

void Init::get_hes_sparsity(FixedVector<int>& i_row_hes, FixedVector<int>& j_col_hes)
{
    for (int nz = 0; nz < formulation().hes_rows.int_size(); nz++) {
        i_row_hes[nz] = formulation().hes_rows[nz];
        j_col_hes[nz] = formulation().hes_cols[nz];
    }
}

void Init::eval_f(bool new_x, const FixedVector<f64>& curr_x, f64& curr_obj)
{
    problem.eval_objective(get_y(curr_x), update_p(curr_x), curr_obj);
}

void Init::eval_g(bool new_x, const FixedVector<f64>& curr_x, FixedVector<f64>& curr_g)
{
    const auto& form = formulation();
    const f64* y = get_y(curr_x);
    const f64* p = update_p(curr_x);

    problem.eval_f(y, p, curr_g.raw());

    if (form.g_size > 0) {
        problem.eval_g(y, p, curr_g.raw() + form.f_size);
    }
}

void Init::eval_grad_f(bool new_x, const FixedVector<f64>& curr_x, FixedVector<f64>& curr_grad_f)
{
    const auto& form = formulation();
    problem.eval_grad_objective(get_y(curr_x), update_p(curr_x), grad_y_buffer.raw(), grad_p_buffer.raw());

    for (int y = 0; y < form.y_size; y++) {
        curr_grad_f[y] = grad_y_buffer[y];
    }

    for (int p = 0; p < form.p_size; p++) {
        curr_grad_f[form.y_size + p] = grad_p_buffer[p];
    }
}

void Init::eval_jac_g(bool new_x,
                      const FixedVector<f64>& curr_x,
                      const FixedVector<int>& i_row_jac,
                      const FixedVector<int>& j_col_jac,
                      FixedVector<f64>& curr_jac)
{
    const auto& form = formulation();
    const f64* y = get_y(curr_x);
    const f64* p = update_p(curr_x);
    int offset = 0;

    problem.eval_jacobian_f(y, p, jac_f_buffer.raw());
    for (int nz = 0; nz < jac_f_buffer.int_size(); nz++) {
        curr_jac[offset++] = jac_f_buffer[nz];
    }

    if (form.g_size > 0) {
        problem.eval_jacobian_g(y, p, jac_g_buffer.raw());
        for (int nz = 0; nz < jac_g_buffer.int_size(); nz++) {
            curr_jac[offset++] = jac_g_buffer[nz];
        }
    }
}

void Init::eval_hes(bool new_x,
                    const FixedVector<f64>& curr_x,
                    bool new_lambda,
                    const FixedVector<f64>& curr_lambda,
                    f64 curr_obj_factor,
                    const FixedVector<int>& i_row_hes,
                    const FixedVector<int>& j_col_hes,
                    FixedVector<f64>& curr_hes)
{
    const auto& form = formulation();
    const f64* lambda_f = curr_lambda.raw();
    const f64* lambda_g = form.g_size > 0 ? curr_lambda.raw() + form.f_size : nullptr;

    curr_hes.fill_zero();
    problem.eval_hessian_constraints(get_y(curr_x), update_p(curr_x), lambda_f, lambda_g, curr_hes.raw());

    if (form.objective == Objective::USER) {
        problem.eval_hessian_objective(get_y(curr_x), p_buffer.raw(), curr_obj_factor, curr_hes.raw());
    } else {
        accumulate_objective_hessian(curr_obj_factor, curr_hes);
    }
}

void Init::finalize_solution(::NLP::ReturnCode ret,
                             f64 opt_obj,
                             const FixedVector<f64>& opt_x,
                             const FixedVector<f64>& opt_lambda,
                             const FixedVector<f64>& opt_z_lb,
                             const FixedVector<f64>& opt_z_ub)
{
    const auto& form = formulation();

    result.objective = opt_obj;
    result.y = FixedVector<f64>(form.y_size);
    result.dp = FixedVector<f64>(form.p_size);
    result.p = FixedVector<f64>(form.p_size);
    result.constraints = FixedVector<f64>(form.f_size + form.g_size);
    result.lambda = FixedVector<f64>(opt_lambda);
    result.z_lb = FixedVector<f64>(opt_z_lb);
    result.z_ub = FixedVector<f64>(opt_z_ub);

    for (int y = 0; y < form.y_size; y++) {
        result.y[y] = opt_x[y];
    }

    for (int p = 0; p < form.p_size; p++) {
        result.dp[p] = opt_x[form.y_size + p];
        result.p[p] = p0_or_zero(form, p) + result.dp[p];
    }

    problem.eval_f(result.y.raw(), result.p.raw(), result.constraints.raw());

    if (form.g_size > 0) {
        problem.eval_g(result.y.raw(), result.p.raw(), result.constraints.raw() + form.f_size);
    }

    update_result_diagnostics();
}

const ProblemFormulation& Init::formulation() const
{
    return problem.formulation;
}

int Init::number_vars_internal() const
{
    return formulation().y_size + formulation().p_size;
}

int Init::number_constraints_internal() const
{
    return formulation().f_size + formulation().g_size;
}

const f64* Init::get_y(const FixedVector<f64>& x) const
{
    return x.raw();
}

const f64* Init::get_dp(const FixedVector<f64>& x) const
{
    return x.raw() + formulation().y_size;
}

const f64* Init::update_p(const FixedVector<f64>& x)
{
    const auto& form = formulation();
    const f64* dp = get_dp(x);

    for (int p = 0; p < form.p_size; p++) {
        p_buffer[p] = p0_or_zero(form, p) + dp[p];
    }

    return p_buffer.raw();
}

void Init::validate_formulation() const
{
    const auto& form = formulation();
    bool ok = true;

    ok &= form.y_size >= 0;
    ok &= form.p_size >= 0;
    ok &= form.f_size >= 0;
    ok &= form.g_size >= 0;

    ok &= valid_optional_size("y0", form.y0.int_size(), form.y_size);
    ok &= valid_optional_size("p0", form.p0.int_size(), form.p_size);
    ok &= valid_optional_size("dp0", form.dp0.int_size(), form.p_size);
    ok &= valid_optional_size("y_bounds", form.y_bounds.int_size(), form.y_size);
    ok &= valid_optional_size("p_bounds", form.p_bounds.int_size(), form.p_size);
    ok &= valid_optional_size("dp_bounds", form.dp_bounds.int_size(), form.p_size);
    ok &= valid_optional_size("g_bounds", form.g_bounds.int_size(), form.g_size);
    ok &= valid_optional_size("y_nominal", form.y_nominal.int_size(), form.y_size);
    ok &= valid_optional_size("p_nominal", form.p_nominal.int_size(), form.p_size);
    ok &= valid_optional_size("dp_nominal", form.dp_nominal.int_size(), form.p_size);
    ok &= valid_optional_size("f_nominal", form.f_nominal.int_size(), form.f_size);
    ok &= valid_optional_size("g_nominal", form.g_nominal.int_size(), form.g_size);

    if (form.jac_f_rows.int_size() != form.jac_f_cols.int_size()) {
        Log::error("Init::ProblemFormulation jac_f_rows and jac_f_cols have different sizes.");
        ok = false;
    }

    if (form.jac_g_rows.int_size() != form.jac_g_cols.int_size()) {
        Log::error("Init::ProblemFormulation jac_g_rows and jac_g_cols have different sizes.");
        ok = false;
    }

    if (form.hes_rows.int_size() != form.hes_cols.int_size()) {
        Log::error("Init::ProblemFormulation hes_rows and hes_cols have different sizes.");
        ok = false;
    }

    for (int nz = 0; nz < form.jac_f_rows.int_size(); nz++) {
        ok &= form.jac_f_rows[nz] >= 0 && form.jac_f_rows[nz] < form.f_size;
        ok &= form.jac_f_cols[nz] >= 0 && form.jac_f_cols[nz] < number_vars_internal();
    }

    for (int nz = 0; nz < form.jac_g_rows.int_size(); nz++) {
        ok &= form.jac_g_rows[nz] >= 0 && form.jac_g_rows[nz] < form.g_size;
        ok &= form.jac_g_cols[nz] >= 0 && form.jac_g_cols[nz] < number_vars_internal();
    }

    for (int nz = 0; nz < form.hes_rows.int_size(); nz++) {
        ok &= form.hes_rows[nz] >= 0 && form.hes_rows[nz] < number_vars_internal();
        ok &= form.hes_cols[nz] >= 0 && form.hes_cols[nz] < number_vars_internal();
    }

    if (!ok) {
        Log::error("Invalid Init::ProblemFormulation.");
        abort();
    }
}

void Init::initialize_objective_hessian_structure()
{
    const auto& form = formulation();

    if (form.objective != Objective::LEAST_SQUARE_DEVIATION) {
        objective_hes_indices = FixedVector<int>(0);
        objective_hes_values = FixedVector<f64>(0);
        return;
    }

    int diagonal_count = 0;
    for (int nz = 0; nz < form.hes_rows.int_size(); nz++) {
        const int row = form.hes_rows[nz];
        const int col = form.hes_cols[nz];
        if (row == col && row >= form.y_size) {
            diagonal_count++;
        }
    }

    objective_hes_indices = FixedVector<int>(diagonal_count);
    objective_hes_values = FixedVector<f64>(diagonal_count);

    if (diagonal_count != form.p_size && !form.hes_rows.empty()) {
        Log::warning("Init::ProblemFormulation least-square objective Hessian has {} parameter diagonal entries, expected {}.",
                     diagonal_count,
                     form.p_size);
    }

    int index = 0;
    for (int nz = 0; nz < form.hes_rows.int_size(); nz++) {
        const int row = form.hes_rows[nz];
        const int col = form.hes_cols[nz];
        if (row == col && row >= form.y_size) {
            const int p_index = row - form.y_size;
            const f64 nominal = nominal_or_one(form.p_nominal, p_index);

            objective_hes_indices[index] = nz;
            objective_hes_values[index] = 2.0 / (nominal * nominal);
            index++;
        }
    }
}

void Init::accumulate_objective_hessian(f64 obj_factor, FixedVector<f64>& curr_hes)
{
    for (int i = 0; i < objective_hes_indices.int_size(); i++) {
        curr_hes[objective_hes_indices[i]] += obj_factor * objective_hes_values[i];
    }
}

void Init::update_result_diagnostics()
{
    const auto& form = formulation();

    result.f_l2_norm = 0.0;
    result.f_max_error = 0.0;
    result.g_max_violation = 0.0;

    for (int f = 0; f < form.f_size; f++) {
        const f64 error = std::abs(result.constraints[f]);
        result.f_l2_norm += error * error;
        result.f_max_error = std::max(result.f_max_error, error);
    }

    result.f_l2_norm = std::sqrt(result.f_l2_norm);

    for (int g = 0; g < form.g_size; g++) {
        const Bounds bounds = bounds_or_unbounded(form.g_bounds, g);
        const f64 value = result.constraints[form.f_size + g];
        f64 violation = 0.0;

        if (bounds.has_lower()) {
            violation = std::max(violation, bounds.lb - value);
        }
        if (bounds.has_upper()) {
            violation = std::max(violation, value - bounds.ub);
        }

        result.g_max_violation = std::max(result.g_max_violation, violation);
    }
}

} // namespace Init
