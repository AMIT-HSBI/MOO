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

constexpr int SPECIES = 250;
constexpr int REACTIONS = SPECIES - 2;
constexpr f64 PARAMETER_PERTURBATION = 1e-1;

f64 measured_parameter(int i)
{
    return 1.0 - PARAMETER_PERTURBATION * std::sin(static_cast<f64>(i + 1) / static_cast<f64>(SPECIES));
}

f64 phase(int i)
{
    return static_cast<f64>(i + 1) / static_cast<f64>(SPECIES);
}

f64 distorted_parameter(int i, f64 p)
{
    const f64 x = p - 1.0;
    const f64 trap = std::sin(36.0 * x + 8.0 * phase(i));

    return 1.0 + x * (1.0 + 2.0 * x * x + 0.25 * trap * trap);
}

f64 d_distorted_parameter_dp(int i, f64 p)
{
    const f64 x = p - 1.0;
    const f64 angle = 36.0 * x + 8.0 * phase(i);
    const f64 trap = std::sin(angle);
    const f64 d_trap = 36.0 * std::cos(angle);

    return 1.0 + 6.0 * x * x + 0.25 * trap * trap + 0.5 * x * trap * d_trap;
}

std::string vector_sample(const FixedVector<f64>& values)
{
    return fmt::format("[{}, {}, {}, ..., {}]",
                       values[0],
                       values[1],
                       values[2],
                       values[values.int_size() - 1]);
}

class ChemicalEquilibriumChain : public Init::Problem {
public:
    ChemicalEquilibriumChain()
    {
        formulation.y_size = SPECIES;
        formulation.p_size = REACTIONS;
        formulation.f_size = REACTIONS;
        formulation.g_size = 0;
        formulation.objective = Init::Objective::LEAST_SQUARE_DEVIATION;

        formulation.y0 = FixedVector<f64>(SPECIES);
        formulation.y_bounds = FixedVector<Bounds>(SPECIES);
        formulation.p0 = FixedVector<f64>(REACTIONS);
        formulation.dp0 = FixedVector<f64>(REACTIONS);
        formulation.p_bounds = FixedVector<Bounds>(REACTIONS);
        formulation.dp_bounds = FixedVector<Bounds>(REACTIONS);
        formulation.p_nominal = FixedVector<f64>(REACTIONS);
        formulation.jac_f_rows = FixedVector<int>(4 * REACTIONS);
        formulation.jac_f_cols = FixedVector<int>(4 * REACTIONS);

        for (int i = 0; i < SPECIES; i++) {
            formulation.y0[i] = 1.0;
            formulation.y_bounds[i] = Bounds{1.0, 1.0};
        }

        for (int r = 0; r < REACTIONS; r++) {
            formulation.p0[r] = measured_parameter(r);
            formulation.dp0[r] = 0.0;
            formulation.p_bounds[r] = Bounds{0.5, 1.5};
            formulation.dp_bounds[r] = Bounds{};
            formulation.p_nominal[r] = 1.0;
        }

        int nz = 0;
        for (int r = 0; r < REACTIONS; r++) {
            formulation.jac_f_rows[nz] = r;
            formulation.jac_f_cols[nz++] = r;
            formulation.jac_f_rows[nz] = r;
            formulation.jac_f_cols[nz++] = r + 1;
            formulation.jac_f_rows[nz] = r;
            formulation.jac_f_cols[nz++] = r + 2;
            formulation.jac_f_rows[nz] = r;
            formulation.jac_f_cols[nz++] = SPECIES + r;
        }
    }

    void eval_f(const f64* c, const f64* p, f64* f) override
    {
        for (int r = 0; r < REACTIONS; r++) {
            f[r] = c[r] * c[r + 1] - distorted_parameter(r, p[r]) * c[r + 2];
        }
    }

    void eval_jacobian_f(const f64* c, const f64* p, f64* jac_f_values) override
    {
        int nz = 0;

        for (int r = 0; r < REACTIONS; r++) {
            jac_f_values[nz++] = c[r + 1];
            jac_f_values[nz++] = c[r];
            jac_f_values[nz++] = -distorted_parameter(r, p[r]);
            jac_f_values[nz++] = -d_distorted_parameter_dp(r, p[r]) * c[r + 2];
        }
    }
};

Init::Result solve(ChemicalEquilibriumChain& problem)
{
    char argv0[] = "test_init_benchmark";
    char* argv[] = {argv0};

    NLP::NLPSolverSettings settings(1, argv);
    settings.set(NLP::Option::Hessian, NLP::HessianOption::LBFGS);
    settings.set(NLP::Option::Tolerance, 1e-12);
    settings.set(NLP::Option::Iterations, 1000);
    settings.set(NLP::Option::CPUTime, 60.0);

    Init::Init init(problem);
    IpoptSolver::IpoptSolver solver(init, settings);
    solver.optimize();

    return init.get_result();
}

void finalize_solution_parameter_correction(const Init::Result& result)
{
    Log::info("\nInit benchmark: nonlinear parameter-correction chemical-equilibrium chain");
    Log::info("  f_max_error = {}", result.f_max_error);
    Log::info("  c* sample   = {}", vector_sample(result.y));
    Log::info("  p* sample   = {}", vector_sample(result.p));
    Log::info("  dp* sample  = {}", vector_sample(result.dp));

    assert(result.f_max_error < 1e-6);

    for (int i = 0; i < SPECIES; i++) {
        assert(std::abs(result.y[i] - 1.0) < 1e-5);
    }

    for (int r = 0; r < REACTIONS; r++) {
        assert(std::abs(result.p[r] - 1.0) < 1e-5);
        assert(std::abs(result.dp[r] - PARAMETER_PERTURBATION * std::sin(phase(r))) < 1e-5);
    }
}

} // namespace

int main()
{
    ChemicalEquilibriumChain correction_problem;
    Init::Result correction_result = solve(correction_problem);
    finalize_solution_parameter_correction(correction_result);

    return 0;
}
