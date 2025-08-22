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

#ifndef MOO_SIMULATION_BUILDER_H
#define MOO_SIMULATION_BUILDER_H

#include <vector>

#include <simulation/integrator/integrator.h>

namespace Simulation {

template <typename BuilderImpl, typename IntegratorImpl>
class IntegratorBuilder {
public:
    virtual ~IntegratorBuilder() = default;

    IntegratorBuilder(ODEFunction ode_func_, f64* x_start_values_, int x_size_)
        : ode_func(ode_func_), x_start_values(x_start_values_), x_size(x_size_) {}

    BuilderImpl& interval(f64 t0_, f64 tf_, int steps_) {
        if (steps_ <= 0 || tf_ <= t0_) LOG_ERROR("Invalid interval");

        t0 = t0_;
        tf = tf_;
        num_steps = steps_;

        dense_output_grid = std::vector<f64>(num_steps + 1);
        f64 dt = (tf - t0) / num_steps;
        for (int i = 0; i <= num_steps; i++) dense_output_grid[i] = t0 + i * dt;

        return static_cast<BuilderImpl&>(*this);
    }

    BuilderImpl& grid(const std::vector<f64> dense_output_grid_) {
        dense_output_grid = dense_output_grid_;
        return static_cast<BuilderImpl&>(*this);
    }

    BuilderImpl& userdata(void* user_data_) {
        user_data = user_data_;
        return static_cast<BuilderImpl&>(*this);
    }

    BuilderImpl& params(f64* parameters_, int p_size_) {
        parameters = parameters_;
        p_size = p_size_;
        return static_cast<BuilderImpl&>(*this);
    }

    BuilderImpl& control(ControlTrajectory* controls_) {
        controls = controls_;
        return static_cast<BuilderImpl&>(*this);
    }

    BuilderImpl& jacobian(JacobianFunction jac_func_,
                          JacobianFormat jac_fmt_,
                          int* i_row_ = nullptr,
                          int* j_col_ = nullptr,
                          int nnz_ = 0) {
        jac_func = jac_func_;
        jac_fmt = jac_fmt_;
        i_row = i_row_;
        j_col = j_col_;
        nnz = nnz_;
        return static_cast<BuilderImpl&>(*this);
    }

    virtual IntegratorImpl build() const = 0;

protected:
    ODEFunction ode_func;
    f64* x_start_values = nullptr;
    int x_size = 0;

    std::vector<f64> dense_output_grid{};

    f64 t0 = 0.0;
    f64 tf = 0.0;
    int num_steps = 1;

    void* user_data = nullptr;
    f64* parameters = nullptr;
    int p_size = 0;

    ControlTrajectory* controls = nullptr;

    JacobianFunction jac_func = nullptr;
    JacobianFormat jac_fmt = JacobianFormat::DENSE;
    int* i_row = nullptr;
    int* j_col = nullptr;
    int nnz = 0;
};

} // namespace Simulation

#endif // MOO_SIMULATION_BUILDER_H
