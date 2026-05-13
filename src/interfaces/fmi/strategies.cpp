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

#include <base/log.h>
#include <interfaces/fmi/strategies.h>

#include <unordered_map>

namespace FMI {

std::shared_ptr<NLP::Scaling> NominalScalingFactory::operator()(const GDOP::GDOP& gdop) {
    // x, g, f of the NLP { min f(x) s.t. g_l <= g(x) <= g_l }
    auto x_nominal = FixedVector<f64>(gdop.get_number_vars());
    auto g_nominal = FixedVector<f64>(gdop.get_number_constraints());
    f64  f_nominal = 1;

    // get problem sizes
    auto x_size  = gdop.get_problem().pc->x_size;
    auto u_size  = gdop.get_problem().pc->u_size;
    auto xu_size = x_size + u_size;
    auto f_size  = gdop.get_problem().pc->x_size;
    auto g_size  = gdop.get_problem().pc->g_size;
    auto r_size  = gdop.get_problem().pc->r_size;
    auto fg_size = f_size + g_size;

    auto has_mayer = gdop.get_problem().pc->has_mayer;
    auto has_lagrange = gdop.get_problem().pc->has_lagrange;

    // create simple lookup: vref -> nominal if exists
    auto map = std::unordered_map<uint32_t, f64>{};
    for (const auto& elem : fmi_data.settings.nominals)
    {
        map[elem.vref] = elem.nominal;
    }

    auto access = [&](uint32_t vref) {
        auto it = map.find(vref);
        return (it != map.end()) ? it->second : 1.0;
    };

    // add the others
    auto xuz_vrefs = fmi_data.get_xuz_vrefs();

    // TODO: not used and not supported yet
    auto p_vrefs = fmi_data.get_p_vrefs();

    f64 DUMMY_NOMINAL = 1.0;

    if (has_mayer && has_lagrange) {
        const f64 nominal_mayer = DUMMY_NOMINAL;    // TODO: what if objective is not set from output?
        const f64 nominal_lagrange = DUMMY_NOMINAL; // same here
        f_nominal = (nominal_mayer + nominal_lagrange) / 2;
    }
    else if (has_lagrange) {
        f_nominal = DUMMY_NOMINAL;
    }
    else if (has_mayer) {
        f_nominal = DUMMY_NOMINAL;
    }

    // (x, u)_(t_node)
    for (int node = 0; node < 1 + gdop.get_mesh().node_count; node++) {
        for (int x = 0; x < x_size; x++) {
            x_nominal[node * xu_size + x] = access(xuz_vrefs[x]);
        }

        for (int u = 0; u < u_size; u++) {
            x_nominal[node * xu_size + x_size + u] = access(xuz_vrefs[u + x_size]);
        }
    }

    for (int node = 0; node < gdop.get_mesh().node_count; node++) {
        for (int f = 0; f < f_size; f++) {
            g_nominal[node * fg_size + f] = x_nominal[f]; // reuse x nominal for dynamic for now!
        }

        for (int g = 0; g < g_size; g++) {
            g_nominal[f_size + node * fg_size + g] = DUMMY_NOMINAL;
        }
    }

    for (int r = 0; r < r_size; r++) {
        g_nominal[gdop.get_off_fg_total() + r] = DUMMY_NOMINAL;
    }

    // artificial constraints are O(u)
    for (int u = 0; u < u_size; u++) {
        g_nominal[gdop.get_off_fgr_total() + u] = DUMMY_NOMINAL;
    }

    return std::make_shared<NLP::NominalScaling>(std::move(x_nominal), std::move(g_nominal), f_nominal);
}

} // namespace FMI
