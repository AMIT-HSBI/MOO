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

#ifndef MOO_FMI_STRATEGIES_H
#define MOO_FMI_STRATEGIES_H

#include <base/util.h>
#include <base/export.h>

#include <nlp/instances/gdop/strategies.h>

#include <interfaces/fmi/problem.h>

namespace FMI {

class NominalScalingFactory : public GDOP::ScalingFactory {
public:
    FMIData& fmi_data;

    NominalScalingFactory(FMIData& fmi_data_) : fmi_data(fmi_data_) {};

    std::shared_ptr<NLP::Scaling> operator()(const GDOP::GDOP& gdop) override;
};

} // namespace FMI

#endif // MOO_FMI_STRATEGIES_H
