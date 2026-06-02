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

#include <interfaces/fmi/problem.h>

int main() {

    FMI::FMISettings settings;
    settings.path = CRANE_FMU_PATH;
    settings.modelname = "Crane";
    settings.t0 = 0.0;
    settings.tf = 5.0;
    settings.intervals = 50;

    // optimize these parameters
    settings.control_vrefs.push_back( { 620756992, -3, 3 } );
    settings.control_vrefs.push_back( { 620756993, -3, 3 } );

    settings.parameter_vrefs = {};

    uint32_t mayer = 603979777;
    uint32_t lagrange = 603979776;
    //settings.mayer_vref = &mayer;
    settings.lagrange_vref = &lagrange;

    settings.path_constraint_vrefs.push_back( { mayer, -5, 5 } );
    //settings.final_constraint_vrefs.push_back( { mayer, -0, 0 } );

    // settings.initial_constraint_vrefs.push_back( { lagrange, -1, 1 } );
    //settings.initial_constraint_vrefs.push_back( { mayer, -1, 1 } );

    FMI::main_fmi(settings);

    return EXIT_SUCCESS;
}
