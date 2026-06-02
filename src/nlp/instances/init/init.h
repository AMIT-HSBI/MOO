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

#ifndef MOO_INIT_H
#define MOO_INIT_H

#include <memory>

#include <nlp/nlp.h>

#include "problem.h"

namespace Init {

class MOO_EXPORT Init : public ::NLP::NLP {
public:
    explicit Init(Problem& problem);

    const Result& get_result() const;

    std::shared_ptr<::NLP::Scaling> get_scaling() override;

    void get_sizes(int& number_vars,
                   int& number_constraints) override;

    void get_bounds(FixedVector<f64>& x_lb,
                    FixedVector<f64>& x_ub,
                    FixedVector<f64>& g_lb,
                    FixedVector<f64>& g_ub) override;

    void get_initial_guess(bool init_x,
                           FixedVector<f64>& x_init,
                           bool init_lambda,
                           FixedVector<f64>& lambda_init,
                           bool init_z,
                           FixedVector<f64>& z_lb_init,
                           FixedVector<f64>& z_ub_init) override;

    void get_nnz(int& nnz_jac,
                 int& nnz_hes) override;

    void get_jac_sparsity(FixedVector<int>& i_row_jac,
                          FixedVector<int>& j_col_jac) override;

    void get_hes_sparsity(FixedVector<int>& i_row_hes,
                          FixedVector<int>& j_col_hes) override;

    void eval_f(bool new_x,
                const FixedVector<f64>& curr_x,
                f64& curr_obj) override;

    void eval_g(bool new_x,
                const FixedVector<f64>& curr_x,
                FixedVector<f64>& curr_g) override;

    void eval_grad_f(bool new_x,
                     const FixedVector<f64>& curr_x,
                     FixedVector<f64>& curr_grad_f) override;

    void eval_jac_g(bool new_x,
                    const FixedVector<f64>& curr_x,
                    const FixedVector<int>& i_row_jac,
                    const FixedVector<int>& j_col_jac,
                    FixedVector<f64>& curr_jac) override;

    void eval_hes(bool new_x,
                  const FixedVector<f64>& curr_x,
                  bool new_lambda,
                  const FixedVector<f64>& curr_lambda,
                  f64 curr_obj_factor,
                  const FixedVector<int>& i_row_hes,
                  const FixedVector<int>& j_col_hes,
                  FixedVector<f64>& curr_hes) override;

    void finalize_solution(::NLP::ReturnCode ret,
                           f64 opt_obj,
                           const FixedVector<f64>& opt_x,
                           const FixedVector<f64>& opt_lambda,
                           const FixedVector<f64>& opt_z_lb,
                           const FixedVector<f64>& opt_z_ub) override;

private:
    Problem& problem;
    Result result;

    FixedVector<f64> p_buffer;
    FixedVector<f64> grad_y_buffer;
    FixedVector<f64> grad_p_buffer;
    FixedVector<f64> jac_f_buffer;
    FixedVector<f64> jac_g_buffer;
    FixedVector<int> objective_hes_indices;
    FixedVector<f64> objective_hes_values;

    const ProblemFormulation& formulation() const;
    int number_vars_internal() const;
    int number_constraints_internal() const;

    const f64* get_y(const FixedVector<f64>& x) const;
    const f64* get_dp(const FixedVector<f64>& x) const;
    const f64* update_p(const FixedVector<f64>& x);

    void validate_formulation() const;
    void initialize_objective_hessian_structure();
    void accumulate_objective_hessian(f64 obj_factor, FixedVector<f64>& curr_hes);
    void update_result_diagnostics();
};

} // namespace Init

#endif // MOO_INIT_H
