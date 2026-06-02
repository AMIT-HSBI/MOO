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

#include <cassert>
#include <cmath>
#include <string>

#include <base/log.h>
#include <nlp/instances/init/init.h>
#include <nlp/solvers/ipopt/solver.h>
#include <nlp/solvers/nlp_solver_settings.h>

namespace {

std::string vector_string(const FixedVector<f64>& values)
{
    std::string out = "[";
    for (int i = 0; i < values.int_size(); i++) {
        out += fmt::format("{}", values[i]);
        if (i < values.int_size() - 1) {
            out += ", ";
        }
    }
    out += "]";
    return out;
}

class TwoParameterInitProblem : public Init::Problem {
public:
    explicit TwoParameterInitProblem(Init::Objective objective)
    {
        formulation.y_size = 2;
        formulation.p_size = 2;
        formulation.f_size = 2;
        formulation.g_size = 0;
        formulation.objective = objective;

        formulation.y0 = FixedVector<f64>{1.0, 0.0};
        formulation.p0 = FixedVector<f64>{0.5, 0.5};
        formulation.dp0 = FixedVector<f64>{0.0, 0.0};
        formulation.y_bounds = FixedVector<Bounds>{Bounds{-10.0, 10.0}, Bounds{-10.0, 10.0}};
        formulation.p_bounds = FixedVector<Bounds>{Bounds{-1.0, 1.0}, Bounds{-1.0, 1.0}};
        formulation.dp_bounds = FixedVector<Bounds>{Bounds{0.0, 0.0}, Bounds{0.0, 0.0}};

        if (objective != Init::Objective::ZERO) {
            formulation.dp_bounds = FixedVector<Bounds>{Bounds{-1.0, 1.0}, Bounds{-1.0, 1.0}};
        }

        formulation.jac_f_rows = FixedVector<int>{0, 0, 1, 1};
        formulation.jac_f_cols = FixedVector<int>{0, 2, 1, 3};
    }

    void eval_objective(const f64* y, const f64* p, f64& obj) override
    {
        if (formulation.objective != Init::Objective::USER) {
            Init::Problem::eval_objective(y, p, obj);
            return;
        }

        obj = (p[0] - formulation.p0[0]) * (p[0] - formulation.p0[0]);
    }

    void eval_grad_objective(const f64* y, const f64* p, f64* grad_y, f64* grad_p) override
    {
        if (formulation.objective != Init::Objective::USER) {
            Init::Problem::eval_grad_objective(y, p, grad_y, grad_p);
            return;
        }

        grad_y[0] = 0.0;
        grad_y[1] = 0.0;
        grad_p[0] = 2.0 * (p[0] - formulation.p0[0]);
        grad_p[1] = 0.0;
    }

    void eval_f(const f64* y, const f64* p, f64* f) override
    {
        f[0] = y[0] * y[0] + p[0];
        f[1] = y[1] * y[1] + p[1];
    }

    void eval_jacobian_f(const f64* y, const f64*, f64* jac_f_values) override
    {
        jac_f_values[0] = 2.0 * y[0];
        jac_f_values[1] = 1.0;
        jac_f_values[2] = 2.0 * y[1];
        jac_f_values[3] = 1.0;
    }
};

Init::Result solve(TwoParameterInitProblem& problem)
{
    char argv0[] = "test_init";
    char* argv[] = {argv0};

    NLP::NLPSolverSettings settings(1, argv);
    settings.set(NLP::Option::Hessian, NLP::HessianOption::LBFGS);
    settings.set(NLP::Option::Tolerance, 1e-9);
    settings.set(NLP::Option::Iterations, 200);
    settings.set(NLP::Option::CPUTime, 20.0);

    Init::Init init(problem);
    IpoptSolver::IpoptSolver solver(init, settings);
    solver.optimize();

    return init.get_result();
}

void print_solution(const char* label, const Init::Result& result)
{
    Log::info("\nInit objective test: {}", label);
    Log::info("  y*  = {}", vector_string(result.y));
    Log::info("  p*  = {}", vector_string(result.p));
    Log::info("  dp* = {}", vector_string(result.dp));
}

void finalize_solution_zero(const Init::Result& result)
{
    print_solution("ZERO", result);
    assert(result.f_max_error > 1e-3);
}

void finalize_solution_user(const Init::Result& result)
{
    print_solution("USER", result);
    assert(result.f_max_error < 1e-6);
    assert(std::abs(result.p[0]) < 1e-4);
    assert(result.p[1] < 1e-4);
}

void finalize_solution_least_square(const Init::Result& result)
{
    print_solution("LEAST_SQUARE_DEVIATION", result);
    assert(result.f_max_error < 1e-6);
    assert(std::abs(result.p[0]) < 1e-4);
    assert(std::abs(result.p[1]) < 1e-4);
}

} // namespace

int main()
{
    TwoParameterInitProblem zero_problem(Init::Objective::ZERO);
    Init::Result zero_result = solve(zero_problem);
    finalize_solution_zero(zero_result);

    TwoParameterInitProblem user_problem(Init::Objective::USER);
    Init::Result user_result = solve(user_problem);
    finalize_solution_user(user_result);

    TwoParameterInitProblem least_square_problem(Init::Objective::LEAST_SQUARE_DEVIATION);
    Init::Result least_square_result = solve(least_square_problem);
    finalize_solution_least_square(least_square_result);

    return 0;
}
