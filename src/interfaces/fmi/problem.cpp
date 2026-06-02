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
#include <interfaces/fmi/problem.h>
#include <interfaces/fmi/expressions.h>
#include <interfaces/fmi/strategies.h>

#include <nlp/instances/gdop/orchestrator.h>
#include <nlp/solvers/ipopt/solver.h>
#include <nlp/instances/gdop/strategies.h>

#include <fmi4c.h>
#include <fmi4c_lsdae.h>

namespace FMI {

enum class VarCategory {
    State,       // x
    Input,       // u
    Algebraic,   // z
    Parameter,   // p
    InitialTime, // t0
    FinalTime,   // tf
    Unknown
};

struct VarIndex {
    VarCategory category;
    int local_index;  // index inside strictly x, u, or z
    int semi_local;   // index inside x or concatenated (u+z)
    int xu_index;     // index inside fully concatenated (x, u+z)
};

struct Dependency {
    fmi3ValueReference vref;
    VarIndex var;
};

// FMIData_priv is split into two layers:
//
//  Layer 1: raw FMU model information, everything needed to classify variables and look up,
//           value references from the FMU XMLs.
//
//  Layer 2: NLP problem structures, exactly the packed vectors and dependency lists that
//           are handed to the MOO library. Built from Layer 1 during construction and never changed afterwards.

struct FMIData_priv {
    fmuHandle*          fmu;
    fmi3InstanceHandle* instance;

    // Layer 1: raw FMU variable sets (disjoint, populated first)

    std::vector<fmi3ValueReference> state_vrefs;     // x
    std::vector<fmi3ValueReference> deriv_vrefs;     // der(x)
    std::vector<fmi3ValueReference> input_vrefs;     // u  (causality == input)
    std::vector<fmi3ValueReference> alg_vrefs;       // z  (fmi-ls-dae algebraic)
    std::vector<fmi3ValueReference> res_vrefs;       // model residuals g = 0
    std::vector<fmi3ValueReference> output_vrefs;    // all outputs y (superset)
    std::vector<fmi3ValueReference> parameter_vrefs; // tunable parameters p

    // maps every Layer-1 vref to its VarIndex (NLP) object
    std::unordered_map<fmi3ValueReference, VarIndex> vref_map;

    // Layer 2: NLP problem structures

    // [x1,..,xn, u1,..,uk, z1,..,zm]
    std::vector<fmi3ValueReference> xuz_vrefs;

    // TODO: [p1,..,pm]
    std::vector<fmi3ValueReference> p_vrefs;

    // TODO: [t0, tf]
    std::vector<fmi3ValueReference> T_vrefs;

    // pointwise functions: (L, f1,..,fn, g1=0,..,gm=0, c1,..,ck)
    // L   = optional Lagrange integrand  (output vref, may be absent)
    // fi  = state derivatives            (deriv_vrefs in order)
    // gi  = model residuals              (res_vrefs in order, lb=ub=0)
    // ci  = user path constraints        (output vrefs, user bounds)
    std::vector<fmi3ValueReference> Lfg_vrefs;
    std::vector<std::vector<Dependency>> Lfg_dependencies;

    bool has_lagrange = false; // true  => Lfg_vrefs[0] is L

    // boundary functions: (M, rf1,..,rfm) at final time
    // M   = optional Mayer term    (output vref, may be absent)
    // rfi = user final constraints (output vrefs, user bounds)
    std::vector<fmi3ValueReference> Mrf_vrefs;
    std::vector<std::vector<Dependency>> Mrf_dependencies;

    bool has_mayer = false; // true => Mrf_vrefs[0] is M

    // boundary functions: (r0_1,..,r0_m) at initial time (output vrefs, user bounds)
    std::vector<fmi3ValueReference> r0_vrefs;
    std::vector<std::vector<Dependency>> r0_dependencies;

    // will be set when creating the sparsity pattern in MOO structures
    size_t Mr_r0_offset = 0;
    size_t Mr_r0_offset_nnz = 0;

    // bounds objects for (g, c) and (rf, r0)
    std::vector<f64> gc_lb;
    std::vector<f64> gc_ub;
    std::vector<f64> rfr0_lb;
    std::vector<f64> rfr0_ub;

    FMIData_priv(const char* path, const char* modelname);
};

// TODO: add const getters for all that may be used in other places: e.g. for strategies
const std::vector<uint32_t>& FMIData::get_xuz_vrefs() const
{
    return priv->xuz_vrefs;
}

const std::vector<uint32_t>& FMIData::get_p_vrefs() const
{
    return priv->p_vrefs;
}

FMIData_priv::FMIData_priv(const char* path, const char* modelname)
    : fmu(fmi4c_loadFmu(path, modelname)),
      instance(fmi3_instantiateModelExchange(fmu, true, true, nullptr, nullptr))
{}

// resolve a dependency vref through the vref_map, returning Unknown if absent
static Dependency make_dependency(fmi3ValueReference vref,
                                  const std::unordered_map<fmi3ValueReference, VarIndex>& vref_map)
{
    auto it = vref_map.find(vref);
    if (it != vref_map.end()) return {vref, it->second};
    return {vref, {VarCategory::Unknown, -1, -1, -1}};
}

// read dependency list from an fmiLsDaeModelStructureHandle into a Dependency vector
static std::vector<Dependency> read_dependencies(fmiLsDaeModelStructureHandle* h,
                                                 const std::unordered_map<fmi3ValueReference, VarIndex>& vref_map)
{
    int n = fmiLsDae_getNumberOfDependencies(h);
    std::vector<Dependency> deps;
    deps.reserve(n);
    if (n > 0) {
        std::vector<fmi3ValueReference> raw(n);
        fmiLsDae_getDependencies(h, raw.data(), n);
        for (auto vref : raw)
            deps.push_back(make_dependency(vref, vref_map));
    }
    return deps;
}

static std::vector<Dependency> make_dep_from_vref(std::vector<fmi3ValueReference>& vref_deps, const std::unordered_map<fmi3ValueReference, VarIndex>& vref_map)
{
    std::vector<Dependency> deps;

    for (auto& vref : vref_deps)
    {
        deps.push_back(make_dependency(vref, vref_map));
    }

    return deps;
}

// find the output handle for a given vref
static fmiLsDaeModelStructureHandle* find_output_handle(fmuHandle* fmu,
                                                         const std::vector<fmi3ValueReference>& output_vrefs,
                                                         fmi3ValueReference target)
{
    int count = static_cast<int>(output_vrefs.size());
    for (int idx = 0; idx < count; idx++) {
        fmiLsDaeModelStructureHandle* h = fmiLsDae_getOutputByIndex(fmu, idx);
        if (h->valueReference == target) return h;
    }
    return nullptr;
}

void FMIData::print() {
    Log::info("\n=== FMI Model Metadata ===");
    Log::info("FMI version  : {}", static_cast<int>(fmi4c_getFmiVersion(priv->fmu)));
    Log::info("Model name   : {}", fmi3_modelName(priv->fmu));

    // Layer 1 summary
    FixedTableFormat<4> summary_fmt = {{15, 8, 25, 10}, {Align::Center, Align::Center, Align::Center, Align::Center}};
    Log::start_module(summary_fmt, "Layer 1 - FMU Variable Counts & Preview");
    Log::row(summary_fmt, "Category", "Count", "First 2 VRefs", "Symbol");
    Log::dashes(summary_fmt);

    auto get_preview = [](const std::vector<fmi3ValueReference>& vrefs) -> std::string {
        if (vrefs.empty()) return "-";
        std::string s;
        size_t take = std::min<size_t>(vrefs.size(), 2);
        for (size_t i = 0; i < take; i++)
            s += std::to_string(vrefs[i]) + (i < take - 1 ? ", " : "");
        if (vrefs.size() > 2) s += "...";
        return s;
    };

    Log::row(summary_fmt, "States",      priv->state_vrefs.size(),     get_preview(priv->state_vrefs),     "x");
    Log::row(summary_fmt, "Derivatives", priv->deriv_vrefs.size(),     get_preview(priv->deriv_vrefs),     "der(x)");
    Log::row(summary_fmt, "Inputs",      priv->input_vrefs.size(),     get_preview(priv->input_vrefs),     "u");
    Log::row(summary_fmt, "Algebraics",  priv->alg_vrefs.size(),       get_preview(priv->alg_vrefs),       "z");
    Log::row(summary_fmt, "Residuals",   priv->res_vrefs.size(),       get_preview(priv->res_vrefs),       "g");
    Log::row(summary_fmt, "Outputs",     priv->output_vrefs.size(),    get_preview(priv->output_vrefs),    "y");
    Log::row(summary_fmt, "Parameters",  priv->parameter_vrefs.size(), get_preview(priv->parameter_vrefs), "p");
    Log::dashes(summary_fmt);

    // Layer 1 index map
    FixedTableFormat<5> map_fmt = {{15, 10, 10, 12, 12}, {Align::Center, Align::Center, Align::Center, Align::Center, Align::Center}};
    Log::start_module(map_fmt, "Layer 1 - Index Mapping");
    Log::row(map_fmt, "Category", "VRef", "Local (L)", "Semi-Loc (SL)", "Global (G)");
    Log::dashes(map_fmt);

    auto print_map_rows = [&](const std::string& cat, const std::vector<fmi3ValueReference>& vrefs) {
        for (auto vref : vrefs) {
            auto it = priv->vref_map.find(vref);
            if (it != priv->vref_map.end())
                Log::row(map_fmt, cat, vref, it->second.local_index, it->second.semi_local, it->second.xu_index);
        }
    };
    print_map_rows("State (x)",    priv->state_vrefs);
    print_map_rows("Input (u)",    priv->input_vrefs);
    print_map_rows("Algebraic (z)", priv->alg_vrefs);
    print_map_rows("Parameter (p)", priv->parameter_vrefs);
    Log::dashes(map_fmt);

    // Layer 2 sparsity
    FixedTableFormat<3> dep_fmt = {{20, 15, 45},
        {Align::Center, Align::Center, Align::Center}};
    Log::start_module(dep_fmt, "Layer 2 - NLP Sparsity (Global Jacobian Column Indices)");
    Log::row(dep_fmt, "Equation", "VRef", "Global Column Indices (G)");
    Log::dashes(dep_fmt);

    auto print_deps = [&](const std::string& prefix,
                          const std::vector<fmi3ValueReference>& vrefs,
                          const std::vector<std::vector<Dependency>>& deps_list)
    {
        for (size_t i = 0; i < vrefs.size(); i++) {
            std::string indices_str;
            for (const auto& d : deps_list[i]) {
                if (d.var.category == VarCategory::Unknown)
                    indices_str += "??, ";
                else
                    indices_str += std::to_string(d.var.xu_index) + ", ";
            }
            if (!indices_str.empty()) indices_str.erase(indices_str.length() - 2);
            Log::row(dep_fmt, fmt::format("{}[{}]", prefix, i), vrefs[i],
                     indices_str.empty() ? "none" : indices_str);
        }
    };

    print_deps("Lfg", priv->Lfg_vrefs, priv->Lfg_dependencies);
    Log::dashes(dep_fmt);

    print_deps("Mrf", priv->Mrf_vrefs, priv->Mrf_dependencies);
    Log::dashes(dep_fmt);

    print_deps("r0", priv->r0_vrefs, priv->r0_dependencies);
    Log::dashes(dep_fmt);
}

FMIData::FMIData(FMISettings& settings)
    : settings(settings),
      priv(std::make_unique<FMIData_priv>(settings.path, settings.modelname))
{
    if (priv->fmu == nullptr) {
        Log::error("fmi4c_loadFmu failed");
        throw std::runtime_error("Failed to load FMU");
    }

    bool with_ls_dae = true;

    if (!fmiLsDae_isPresent(priv->fmu)) {
        Log::info("fmi-ls-dae manifest not found in FMU");
        with_ls_dae = false;
    }

    // Layer 1: populate raw FMU variable sets
    size_t nStates;

    if (with_ls_dae)
    {
        nStates = fmiLsDae_getNumberOfContinuousStateDerivatives(priv->fmu);
    }
    else
    {
        fmi3_getNumberOfContinuousStates(priv->instance, &nStates);
        Log::info("{} states", nStates);
    }

    // create work buffer with size of states
    work = std::vector<f64>(nStates);

    // states and their derivatives
    {
        if (with_ls_dae)
        {
            for (size_t s = 0; s < nStates; s++) {
                auto* h      = fmiLsDae_getContinuousStateDerivativeByIndex(priv->fmu, s);
                auto der_ref = fmiLsDae_getValueReference(h);
                auto x_ref   = fmi3_getVariableDerivativeIndex(fmi3_getVariableByValueReference(priv->fmu, der_ref));
                priv->state_vrefs.push_back(x_ref);
                priv->deriv_vrefs.push_back(der_ref);
            }
        }
        else
        {
            for (size_t var_idx = 0; var_idx < nStates; var_idx++)
            {
                auto* der_mhandle = fmi3_getModelStructureContinuousStateDerivative(priv->fmu, var_idx);
                auto der_vref = fmi3_getModelStructureValueReference(der_mhandle);
                auto* der_handle = fmi3_getVariableByValueReference(priv->fmu, der_vref);
                auto state_vref = fmi3_getVariableDerivativeIndex(der_handle);
                if (state_vref != 0)
                {
                    priv->state_vrefs.push_back(state_vref);
                    priv->deriv_vrefs.push_back(fmi3_getVariableValueReference(der_handle));
                }
            }
        }
    }

    // inputs
    {
        for (auto ref : settings.control_vrefs) {
            auto* handle = fmi3_getVariableByValueReference(priv->fmu, ref.vref);
            auto causality = fmi3_getVariableCausality(handle);
            if (causality != fmi3CausalityInput) {
                Log::error("Control vref {} is not an input.", ref.vref);
                throw std::runtime_error("Non-input control vref");
            }
            priv->input_vrefs.push_back(ref.vref);
        }
    }

    // algebraic variables
    {
        int n = fmiLsDae_getNumberOfAlgebraicVariables(priv->fmu);
        priv->alg_vrefs.reserve(n);
        for (int i = 0; i < n; i++) {
            auto* h = fmiLsDae_getAlgebraicVariableByIndex(priv->fmu, i);
            priv->alg_vrefs.push_back(fmiLsDae_getAlgebraicVariableValueReference(h));
        }
    }

    // model residuals
    {
        int n = fmiLsDae_getNumberOfResiduals(priv->fmu);
        priv->res_vrefs.reserve(n);
        for (int i = 0; i < n; i++) {
            auto* h = fmiLsDae_getResidualByIndex(priv->fmu, i);
            priv->res_vrefs.push_back(fmiLsDae_getValueReference(h));
        }
    }

    // all outputs (includes: Lagrange, Mayer, path, final, ...)
    {
        int n = fmiLsDae_getNumberOfOutputs(priv->fmu);
        priv->output_vrefs.reserve(n);
        for (int i = 0; i < n; i++) {
            auto* h = fmiLsDae_getOutputByIndex(priv->fmu, i);
            priv->output_vrefs.push_back(fmiLsDae_getValueReference(h));
        }
    }

    // tunable parameters
    for (auto ref : settings.parameter_vrefs) {
        auto* handle     = fmi3_getVariableByValueReference(priv->fmu, ref.vref);
        auto variability = fmi3_getVariableVariability(handle);
        if (variability != fmi3VariabilityTunable) {
            Log::error("Parameter vref {} is not tunable.", ref.vref);
            throw std::runtime_error("Non-tunable parameter vref");
        }
        priv->parameter_vrefs.push_back(ref.vref);
    }

    // validate Lagrange / Mayer / path / final vrefs exist in output_vrefs
    auto require_output = [&](uint32_t vref, const char* label) {
        auto it = std::find(priv->output_vrefs.begin(), priv->output_vrefs.end(),
                            static_cast<fmi3ValueReference>(vref));
        if (it == priv->output_vrefs.end()) {
            Log::error("{} vref {} not found in outputs.", label, vref);
            throw std::runtime_error(std::string(label) + " vref not in outputs");
        }
    };

    if (settings.lagrange_vref) require_output(*settings.lagrange_vref, "Lagrange");
    if (settings.mayer_vref) require_output(*settings.mayer_vref,    "Mayer");
    for (auto& bv : settings.path_constraint_vrefs) require_output(bv.vref, "Path constraint");
    for (auto& bv : settings.final_constraint_vrefs) require_output(bv.vref, "Final constraint");

    // build variable reference map
    {
        int xu_offset = 0;
        int u_size    = static_cast<int>(priv->input_vrefs.size());

        for (size_t i = 0; i < priv->state_vrefs.size(); i++)
            priv->vref_map[priv->state_vrefs[i]] = {
                VarCategory::State, (int)i, (int)i, xu_offset++};

        for (size_t i = 0; i < priv->input_vrefs.size(); i++)
            priv->vref_map[priv->input_vrefs[i]] = {
                VarCategory::Input, (int)i, (int)i, xu_offset++};

        for (size_t i = 0; i < priv->alg_vrefs.size(); i++)
            priv->vref_map[priv->alg_vrefs[i]] = {
                VarCategory::Algebraic, (int)i, u_size + (int)i, xu_offset++};

        for (size_t i = 0; i < priv->parameter_vrefs.size(); i++)
            priv->vref_map[priv->parameter_vrefs[i]] = {
                VarCategory::Parameter, (int)i, (int)i, (int)i};
    }

    // Layer 2: build NLP problem structures

    // variable vectors
    for (auto v : priv->state_vrefs)     priv->xuz_vrefs.push_back(v);
    for (auto v : priv->input_vrefs)     priv->xuz_vrefs.push_back(v);
    for (auto v : priv->alg_vrefs)       priv->xuz_vrefs.push_back(v);
    for (auto v : priv->parameter_vrefs) priv->p_vrefs.push_back(v);

    // Lfg: (L?, f1..fn, g1..gm, c1..ck)

    // L: Lagrange term?
    if (settings.lagrange_vref) {
        priv->has_lagrange = true;
        priv->Lfg_vrefs.push_back(static_cast<fmi3ValueReference>(*settings.lagrange_vref));

        auto* h = find_output_handle(priv->fmu, priv->output_vrefs, priv->Lfg_vrefs.back());
        priv->Lfg_dependencies.push_back(h ? read_dependencies(h, priv->vref_map) : std::vector<Dependency>{});
    }
    else if (!settings.lagrange_expr.empty())
    {
        priv->has_lagrange = true;
        std::vector<Dependency> deps;
        priv->Lfg_vrefs.push_back(static_cast<fmi3ValueReference>(0));
        for (auto const& e : settings.lagrange_expr.terms)
        {
            // create self dependency as function is of form sum a_i * x_i^2 + b_i * x_i + c_i
            deps.push_back(make_dependency(e.vref, priv->vref_map));
        }
        priv->Lfg_dependencies.push_back(deps);
    }

    // f: state derivatives
    {
        if (with_ls_dae)
        {
            int n = static_cast<int>(priv->state_vrefs.size());
            for (int i = 0; i < n; i++) {
                priv->Lfg_vrefs.push_back(priv->deriv_vrefs[i]);
                auto* h = fmiLsDae_getContinuousStateDerivativeByIndex(priv->fmu, i);
                priv->Lfg_dependencies.push_back(h ? read_dependencies(h, priv->vref_map) : std::vector<Dependency>{});
            }
        }
        /*
        else
        {
            for (int i = 0; i < static_cast<int>(priv->state_vrefs.size()); i++) {
                priv->Lfg_vrefs.push_back(priv->deriv_vrefs[i]);
                auto dep_vrefs = get_filtered_dependencies(priv->instance, priv->deriv_vrefs[i]);
                priv->Lfg_dependencies.push_back(make_dep_from_vref(dep_vrefs, priv->vref_map));
            }
        }*/
        else
        {
            std::vector<fmi3ValueReference> dense;
            dense.reserve(priv->state_vrefs.size() + priv->input_vrefs.size());

            for (auto vref : priv->state_vrefs)
            {
                dense.push_back(vref);
            }

            for (auto vref : priv->input_vrefs)
            {
                dense.push_back(vref);
            }

            for (int i = 0; i < static_cast<int>(priv->state_vrefs.size()); i++) {
                priv->Lfg_vrefs.push_back(priv->deriv_vrefs[i]);
                priv->Lfg_dependencies.push_back(make_dep_from_vref(dense, priv->vref_map));
            }
        }
    }

    // g: model residuals (lb = ub = 0)
    {
        int n = static_cast<int>(priv->res_vrefs.size());
        for (int i = 0; i < n; i++) {
            priv->Lfg_vrefs.push_back(priv->res_vrefs[i]);
            auto* h = fmiLsDae_getResidualByIndex(priv->fmu, i);
            priv->Lfg_dependencies.push_back(read_dependencies(h, priv->vref_map));
            priv->gc_lb.push_back(0.0);
            priv->gc_ub.push_back(0.0);
        }
    }

    // c: user path constraints
    for (auto& bv : settings.path_constraint_vrefs) {
        auto vref = static_cast<fmi3ValueReference>(bv.vref);
        priv->Lfg_vrefs.push_back(vref);
        auto* h = find_output_handle(priv->fmu, priv->output_vrefs, vref);
        priv->Lfg_dependencies.push_back(h ? read_dependencies(h, priv->vref_map) : std::vector<Dependency>{});
        priv->gc_lb.push_back(bv.lb);
        priv->gc_ub.push_back(bv.ub);
    }

    // Mrf: (M?, rf1..rfm) + r0: (r01, ..., r0m)

    // M: Mayer term?
    if (settings.mayer_vref) {
        priv->has_mayer = true;
        priv->Mrf_vrefs.push_back(static_cast<fmi3ValueReference>(*settings.mayer_vref));
        auto* h = find_output_handle(priv->fmu, priv->output_vrefs, priv->Mrf_vrefs.back());
        priv->Mrf_dependencies.push_back(h ? read_dependencies(h, priv->vref_map) : std::vector<Dependency>{});
    }

    // rf: user final constraints
    for (auto& bv : settings.final_constraint_vrefs) {
        auto vref = static_cast<fmi3ValueReference>(bv.vref);
        priv->Mrf_vrefs.push_back(vref);
        auto* h = find_output_handle(priv->fmu, priv->output_vrefs, vref);
        priv->Mrf_dependencies.push_back(h ? read_dependencies(h, priv->vref_map) : std::vector<Dependency>{});
        priv->rfr0_lb.push_back(bv.lb);
        priv->rfr0_ub.push_back(bv.ub);
    }

    // r0: user initial constraints
    for (auto& bv : settings.initial_constraint_vrefs) {
        auto vref = static_cast<fmi3ValueReference>(bv.vref);
        priv->r0_vrefs.push_back(vref);
        auto* h = find_output_handle(priv->fmu, priv->output_vrefs, vref);
        priv->r0_dependencies.push_back(h ? read_dependencies(h, priv->vref_map) : std::vector<Dependency>{});
        priv->rfr0_lb.push_back(bv.lb);
        priv->rfr0_ub.push_back(bv.ub);
    }

    print();
}

void FMIData::initialize(f64 t_start, f64 t_stop) {
    fmi3Status status = fmi3_enterInitializationMode(
        priv->instance,
        fmi3False, 0.0,     // toleranceDefined, tolerance
        t_start,            // startTime
        fmi3True, t_stop    // stopTimeDefined, stopTime
    );

    if (status != fmi3OK) {
        throw std::runtime_error("FMU: Failed to enter initialization mode");
    }

    status = fmi3_exitInitializationMode(priv->instance);
    if (status != fmi3OK) {
        throw std::runtime_error("FMU: Failed to exit initialization mode");
    }

    /* TODO: What do we need to do here?

    status = fmi3_enterEventMode(inst);
    if (status != fmi3OK) {
        throw std::runtime_error("FMU: Failed to enter event mode");
    }

    status = fmi3_enterStepMode(inst);
    if (status != fmi3OK) {
        throw std::runtime_error("FMU: Failed to enter step time mode");
    }

    */

    Log::info("FMU is now in Continuous-Time Mode (ready for simulation).");
}

std::vector<f64> FMIData::get_initial_states() {
    if (priv->state_vrefs.empty()) return {};

    std::vector<f64> states(priv->state_vrefs.size());

    fmi3Status status = fmi3_getFloat64(priv->instance,
                                        priv->state_vrefs.data(),
                                        priv->state_vrefs.size(),
                                        states.data(),
                                        states.size());

    if (status != fmi3OK) {
        Log::error("Failed to retrieve initial states from FMU");
    }

    return states;
}

// evaluate (L?, f, g, c)(x, u, z, p, time)
// buffer layout: [L (opt), f0..fn-1, g0..gm-1, c0..ck-1]
// @note: evaluate the model via fmi3_getContinuousStateDerivatives, before using getFloat64
void FMIData::eval_point_lfg(const f64* xu, const f64* p, f64 time, f64* out)
{
    int   n_x    = static_cast<int>(priv->state_vrefs.size());
    int   n_res  = static_cast<int>(priv->res_vrefs.size());
    int   n_path = static_cast<int>(priv->gc_lb.size()) - n_res;
    bool  has_L  = priv->has_lagrange;

    fmi3_setTime(priv->instance, time);
    fmi3_setFloat64(priv->instance,
                    priv->xuz_vrefs.data(), priv->xuz_vrefs.size(),
                    xu, priv->xuz_vrefs.size());

    int off_L  = 0;
    int off_f  = has_L ? 1 : 0;
    int off_gc = off_f + n_x;

    // f, evaluation of model
    fmi3_getContinuousStateDerivatives(priv->instance, out + off_f, n_x);

    // L
    if (has_L && settings.lagrange_expr.empty()) {
        fmi3_getFloat64(priv->instance,
                        priv->Lfg_vrefs.data(), 1,
                        out + off_L, 1);
    }
    else if (!settings.lagrange_expr.empty())
    {
        auto get_ref_value = [&](uint32_t ref) {
            f64 work;
            fmi3_getFloat64(priv->instance, &ref, 1, &work, 1);
            return work;
        };
        out[off_L] = settings.lagrange_expr.eval(time, get_ref_value);
    }

    // g, c
    int n_gc = n_res + n_path;
    if (n_gc > 0) {
        fmi3_getFloat64(priv->instance,
                        priv->Lfg_vrefs.data() + off_gc, n_gc,
                        out + off_gc, n_gc);
    }
}

// evaluate sparse Jacobian of Lfg w.r.t. (x, u, z, p)
// out[] is filled in the same order as Lfg_dependencies
void FMIData::jac_point_lfg(const f64* xu, const f64* p, f64 time, f64* out) {
    fmi3_setTime(priv->instance, time);
    fmi3_setFloat64(priv->instance,
                    priv->xuz_vrefs.data(), priv->xuz_vrefs.size(),
                    xu, priv->xuz_vrefs.size());

    int out_idx = 0;
    auto get_partial = [&](fmi3ValueReference val_vref, fmi3ValueReference indep_vref) -> f64 {
        f64 delta = 1.0, result = 0.0;
        fmi3Status s = fmi3_getDirectionalDerivative(priv->instance,
                                                     &val_vref, 1,
                                                     &indep_vref, 1,
                                                     &delta, 1,
                                                     &result, 1);
        return (s == fmi3OK) ? result : 0.0;
    };

    int row_off = 0;

    if (!settings.lagrange_expr.empty())
    {
        auto get_ref_value = [&](uint32_t ref) {
            f64 work;
            fmi3_getFloat64(priv->instance, &ref, 1, &work, 1);
            return work;
        };

        for (auto& L_i : settings.lagrange_expr.terms)
        {
            out[out_idx++] = settings.lagrange_expr.deval_dy(L_i, get_ref_value(L_i.vref), time);
        }

        row_off++;
    }

    for (size_t row = row_off; row < priv->Lfg_vrefs.size(); row++) {
        fmi3ValueReference row_vref = priv->Lfg_vrefs[row];
        for (const auto& dep : priv->Lfg_dependencies[row]) {
            out[out_idx++] = get_partial(row_vref, dep.vref);
        }
    }
}

// evaluate (M?, rf)(xf, uf, zf, p, tf)
void FMIData::eval_point_mrf(const f64* xuf, const f64* p, f64 tf, f64* out)
{
    fmi3_setTime(priv->instance, tf);
    fmi3_setFloat64(priv->instance,
                    priv->xuz_vrefs.data(), priv->xuz_vrefs.size(),
                    xuf, priv->xuz_vrefs.size());

    // evaluation of model
    fmi3_getContinuousStateDerivatives(priv->instance, work.data(), work.size());

    fmi3_getFloat64(priv->instance,
                    priv->Mrf_vrefs.data(), priv->Mrf_vrefs.size(),
                    out, priv->Mrf_vrefs.size());
}

// evaluate sparse Jacobian of Mrf w.r.t. (xf, uf, zf, p, tf)
void FMIData::jac_point_mrf(const f64* xuf, const f64* p, f64 tf, f64* out)
{
    fmi3_setTime(priv->instance, tf);
    fmi3_setFloat64(priv->instance,
                    priv->xuz_vrefs.data(), priv->xuz_vrefs.size(),
                    xuf, priv->xuz_vrefs.size());

    int out_idx = 0;
    auto get_partial = [&](fmi3ValueReference val_vref, fmi3ValueReference indep_vref) -> f64 {
        f64 delta = 1.0, result = 0.0;
        fmi3Status s = fmi3_getDirectionalDerivative(priv->instance,
                                                     &val_vref, 1,
                                                     &indep_vref, 1,
                                                     &delta, 1,
                                                     &result, 1);
        return (s == fmi3OK) ? result : 0.0;
    };

    for (size_t row = 0; row < priv->Mrf_vrefs.size(); row++) {
        fmi3ValueReference row_vref = priv->Mrf_vrefs[row];
        for (const auto& dep : priv->Mrf_dependencies[row]) {
            out[out_idx++] = get_partial(row_vref, dep.vref);
        }
    }
}

// evaluate r0(x0, u0, z0, p, t0)
void FMIData::eval_point_r0(const f64* xu0, const f64* p, f64 t0, f64* out)
{
    fmi3_setTime(priv->instance, t0);
    fmi3_setFloat64(priv->instance,
                    priv->xuz_vrefs.data(), priv->xuz_vrefs.size(),
                    xu0, priv->xuz_vrefs.size());

    // evaluation of model
    fmi3_getContinuousStateDerivatives(priv->instance, work.data(), work.size());

    fmi3_getFloat64(priv->instance,
                    priv->r0_vrefs.data(), priv->r0_vrefs.size(),
                    out, priv->r0_vrefs.size());
}

// evaluate sparse Jacobian of r0 w.r.t. (x0, u0, z0, p, t0)
void FMIData::jac_point_r0(const f64* xu0, const f64* p, f64 t0, f64* out)
{
    fmi3_setTime(priv->instance, t0);
    fmi3_setFloat64(priv->instance,
                    priv->xuz_vrefs.data(), priv->xuz_vrefs.size(),
                    xu0, priv->xuz_vrefs.size());

    int out_idx = 0;
    auto get_partial = [&](fmi3ValueReference val_vref, fmi3ValueReference indep_vref) -> f64 {
        f64 delta = 1.0, result = 0.0;
        fmi3Status s = fmi3_getDirectionalDerivative(priv->instance,
                                                     &val_vref, 1,
                                                     &indep_vref, 1,
                                                     &delta, 1,
                                                     &result, 1);
        return (s == fmi3OK) ? result : 0.0;
    };

    for (size_t row = 0; row < priv->r0_vrefs.size(); row++) {
        fmi3ValueReference row_vref = priv->r0_vrefs[row];
        for (const auto& dep : priv->r0_dependencies[row]) {
            out[out_idx++] = get_partial(row_vref, dep.vref);
        }
    }
}

GDOP::ProblemConstants create_problem_constants(FMIData& fmi_data)
{
    auto& priv = *fmi_data.priv;

    auto x_bounds = FixedVector<Bounds>(priv.state_vrefs.size());
    auto u_bounds = FixedVector<Bounds>(priv.input_vrefs.size() + priv.alg_vrefs.size());
    auto p_bounds = FixedVector<Bounds>(0);
    auto T_bounds = std::array<Bounds, 2>{};

    auto g_bounds = FixedVector<Bounds>(priv.gc_lb.size());
    for (int i = 0; i < (int)priv.gc_lb.size(); i++) {
        g_bounds[i].lb = priv.gc_lb[i];
        g_bounds[i].ub = priv.gc_ub[i];
    }

    auto r_bounds = FixedVector<Bounds>(priv.rfr0_lb.size());
    for (size_t i = 0; i < priv.rfr0_lb.size(); i++) {
        r_bounds[i].lb = priv.rfr0_lb[i];
        r_bounds[i].ub = priv.rfr0_ub[i];
    }

    auto xu0_fixed = FixedVector<std::optional<f64>>(x_bounds.size() + u_bounds.size());
    auto xuf_fixed = FixedVector<std::optional<f64>>(x_bounds.size() + u_bounds.size());
    auto T_fixed   = std::array<std::optional<f64>, 2>{};

    auto x0 = fmi_data.get_initial_states();
    for (size_t i = 0; i < x0.size(); i++) {
        xu0_fixed[i] = x0[i];
    }

    for (size_t i = 0; i < fmi_data.settings.fixed_start_values.size(); i++) {
        auto start = fmi_data.settings.fixed_start_values[i];
        auto idx = fmi_data.priv->vref_map[start.vref].xu_index;
        xu0_fixed[idx] = start.value;
    }

    for (size_t i = 0; i < fmi_data.settings.control_vrefs.size(); i++) {
        auto obj = fmi_data.settings.control_vrefs[i];
        u_bounds[i].lb = obj.lb;
        u_bounds[i].ub = obj.ub;
    }

    Log::info("\n=== Start values for States x(0) ===");
    for (size_t i = 0; i < x0.size(); i++)
        Log::info("  x[{}](0) = {}", i, x0[i]);

    Log::info("\n=== Problem Bounds ===");
    for (size_t i = 0; i < x_bounds.size(); i++)
        Log::info("  x[{}]: [{}, {}]", i, x_bounds[i].lb, x_bounds[i].ub);
    for (size_t i = 0; i < priv.input_vrefs.size(); i++)
        Log::info("  u[{}]: [{}, {}]", i, u_bounds[i].lb, u_bounds[i].ub);
    for (size_t i = priv.input_vrefs.size(); i < u_bounds.size(); i++)
        Log::info("  z[{}]: [{}, {}]", i - priv.input_vrefs.size(), u_bounds[i].lb, u_bounds[i].ub);
    for (size_t i = 0; i < g_bounds.size(); i++)
        Log::info("  g[{}]: [{}, {}]", i, g_bounds[i].lb, g_bounds[i].ub);
    for (size_t i = 0; i < r_bounds.size(); i++)
        Log::info("  rf[{}]: [{}, {}]", i, r_bounds[i].lb, r_bounds[i].ub);
    Log::info("================================");

    auto mesh = Mesh::create_equidistant_fixed_stages(
        /* t0 */        fmi_data.settings.t0,
        /* tf */        fmi_data.settings.tf,
        /* intervals */ fmi_data.settings.intervals,
        /* stages */    fmi_data.settings.stage,
        /* type */      MeshType::Physical);

    T_fixed[0] = mesh->t0;
    T_fixed[1] = mesh->tf;

    return GDOP::ProblemConstants(
        priv.has_mayer,
        priv.has_lagrange,
        std::move(x_bounds),
        std::move(u_bounds),
        std::move(p_bounds),
        std::move(T_bounds),
        std::move(xu0_fixed),
        std::move(xuf_fixed),
        std::move(T_fixed),
        std::move(r_bounds),
        std::move(g_bounds),
        *mesh);
}

FullSweep::FullSweep(GDOP::FullSweepLayout&& layout_in,
                     const GDOP::ProblemConstants& pc,
                     FMIData& fmi_data)
    : GDOP::FullSweep(std::move(layout_in), pc),
      fmi_data(fmi_data)
{}

BoundarySweep::BoundarySweep(GDOP::BoundarySweepLayout&& layout_in,
                             const GDOP::ProblemConstants& pc,
                             FMIData& fmi_data)
    : GDOP::BoundarySweep(std::move(layout_in), pc),
      fmi_data(fmi_data)
{}

void FullSweep::callback_eval(const f64* xu_nlp, const f64* p)
{
    fill_zero_eval_buffer();
    for (int i = 0; i < pc.mesh->intervals; i++) {
        for (int j = 0; j < pc.mesh->nodes[i]; j++) {
            fmi_data.eval_point_lfg(get_xu_ij(xu_nlp, i, j), p, pc.mesh->t[i][j], get_eval_buffer(i, j));
        }
    }
}

void FullSweep::callback_jac(const f64* xu_nlp, const f64* p)
{
    fill_zero_jac_buffer();
    for (int i = 0; i < pc.mesh->intervals; i++) {
        for (int j = 0; j < pc.mesh->nodes[i]; j++) {
            fmi_data.jac_point_lfg(get_xu_ij(xu_nlp, i, j), p, pc.mesh->t[i][j], get_jac_buffer(i, j));
        }
    }
}

void FullSweep::callback_hes(const f64* xu_nlp, const f64* p,
                             const FixedField<f64, 2>& lagrange_factors,
                             const f64* lambda)
{
    // not yet implemented
}

void BoundarySweep::callback_eval(const f64* x0_nlp, const f64* xuf_nlp,
                                  const f64* p, f64 t0, f64 tf)
{
    fill_zero_eval_buffer();
    fmi_data.eval_point_mrf(xuf_nlp, p, tf, get_eval_buffer());
    fmi_data.eval_point_r0(x0_nlp, p, t0, get_eval_buffer() + fmi_data.priv->Mr_r0_offset);
}

void BoundarySweep::callback_jac(const f64* x0_nlp, const f64* xuf_nlp,
                                 const f64* p, f64 t0, f64 tf)
{
    fill_zero_jac_buffer();
    fmi_data.jac_point_mrf(xuf_nlp, p, tf, get_jac_buffer());
    fmi_data.jac_point_r0(x0_nlp, p, t0, get_jac_buffer() + fmi_data.priv->Mr_r0_offset_nnz);
}

void BoundarySweep::callback_hes(const f64* x0_nlp, const f64* xuf_nlp,
                                 const f64* p, f64 t0, f64 tf,
                                 const f64 mayer_factor, const f64* lambda)
{
    // not yet implemented
}

void layout_lfg_init_eval(GDOP::FullSweepLayout& layout, GDOP::ProblemConstants& pc)
{
    int buf = 0;

    if (layout.L) {
        layout.L->buf_index = buf++;
    }

    for (int i = 0; i < pc.x_size; i++) {
        layout.f[i].buf_index = buf++;
    }

    for (int i = 0; i < pc.g_size; i++) {
        layout.g[i].buf_index = buf++;
    }
}

void layout_lfg_init_jac(GDOP::FullSweepLayout& layout,
                         FMIData& fmi_data,
                         GDOP::ProblemConstants& pc)
{
    auto& priv  = *fmi_data.priv;
    int buf_idx = 0;
    int lfg_row = 0;

    auto push_dep = [&](auto& jac_entry, const Dependency& dep) {
        if (dep.var.category == VarCategory::State) {
            jac_entry.dx.push_back({dep.var.semi_local, buf_idx++});
        }
        else if (dep.var.category == VarCategory::Input || dep.var.category == VarCategory::Algebraic) {
            jac_entry.du.push_back({dep.var.semi_local, buf_idx++});
        }
        else if (dep.var.category == VarCategory::Parameter) {
            jac_entry.dp.push_back({dep.var.local_index, buf_idx++});
        }
        // Unknown / time: skip for now
    };

    if (layout.L) {
        for (const auto& dep : priv.Lfg_dependencies[lfg_row]) {
            push_dep(layout.L->jac, dep);
        }
        lfg_row++;
    }

    for (int i = 0; i < pc.x_size; i++) {
        for (const auto& dep : priv.Lfg_dependencies[lfg_row]) {
            push_dep(layout.f[i].jac, dep);
        }
        lfg_row++;
    }

    for (int i = 0; i < pc.g_size; i++) {
        for (const auto& dep : priv.Lfg_dependencies[lfg_row]) {
            push_dep(layout.g[i].jac, dep);
        }
        lfg_row++;
    }
    Log::info("FullSweep Jacobian sparsity: total NNZ per node = {}", buf_idx);
}

void layout_mrf_init_eval(GDOP::BoundarySweepLayout& layout)
{
    int buf = 0;

    if (layout.M)
        layout.M->buf_index = buf++;

    for (size_t i = 0; i < layout.r.size(); i++)
        layout.r[i].buf_index = buf++;
}

void layout_mrf_init_jac(GDOP::BoundarySweepLayout& layout,
                          FMIData& fmi_data,
                          GDOP::ProblemConstants& pc)
{
    auto& priv  = *fmi_data.priv;
    int buf_idx = 0;
    int mr_row  = 0;

    auto push_dep = [&](auto& jac_entry, const Dependency& dep, bool final) {
        auto& dx = (final ? jac_entry.dxf : jac_entry.dx0);
        auto& du = (final ? jac_entry.duf : jac_entry.du0);
        auto& dp = jac_entry.dp;
        int time_index = (final ? 1 : 0);

        if (dep.var.category == VarCategory::State) {
            dx.push_back({dep.var.semi_local, buf_idx++});
        }
        else if (dep.var.category == VarCategory::Input || dep.var.category == VarCategory::Algebraic) {
            du.push_back({dep.var.semi_local, buf_idx++});
        }
        else if (dep.var.category == VarCategory::Parameter) {
            dp.push_back({dep.var.local_index, buf_idx++});
        }
        else if (dep.var.category == VarCategory::FinalTime || dep.var.category == VarCategory::InitialTime) {
            jac_entry.dT.push_back({time_index, buf_idx++});
        }
        // Unknown / InitialTime: skip for now
    };

    if (layout.M) {
        for (const auto& dep : priv.Mrf_dependencies[mr_row]) {
            push_dep(layout.M->jac, dep, true);
        }
        mr_row++;
    }

    size_t Mr_M_offset = mr_row;

    for (size_t i = 0; i < priv.Mrf_vrefs.size() - Mr_M_offset; i++) {
        for (const auto& dep : priv.Mrf_dependencies[i + Mr_M_offset]) {
            push_dep(layout.r[i].jac, dep, true);
        }
        mr_row++;
    }

    // jit: set the offset for r0, that are required for the callback implementations
    fmi_data.priv->Mr_r0_offset = mr_row;
    fmi_data.priv->Mr_r0_offset_nnz = buf_idx;

    size_t r_internal_offset = fmi_data.priv->Mr_r0_offset - Mr_M_offset;

    for (size_t i = 0; i < priv.r0_vrefs.size(); i++) {
        for (const auto& dep : priv.r0_dependencies[i]) {
            push_dep(layout.r[r_internal_offset + i].jac, dep, false);
        }
        mr_row++;
    }

    Log::info("BoundarySweep Jacobian sparsity: total NNZ = {}", buf_idx);
}

GDOP::FullSweepLayout create_fullsweep_layout(FMIData& fmi_data, GDOP::ProblemConstants& pc)
{
    auto layout = GDOP::FullSweepLayout(pc.has_lagrange, pc.x_size, pc.g_size);
    layout_lfg_init_eval(layout, pc);
    layout_lfg_init_jac(layout, fmi_data, pc);
    return layout;
}

GDOP::BoundarySweepLayout create_boundarysweep_layout(FMIData& fmi_data, GDOP::ProblemConstants& pc)
{
    auto layout = GDOP::BoundarySweepLayout(pc.has_mayer, pc.r_size);
    layout_mrf_init_eval(layout);
    layout_mrf_init_jac(layout, fmi_data, pc);
    return layout;
}

GDOP::Problem create_gdop_problem(FMIData& fmi_data)
{
    auto pc = std::make_unique<GDOP::ProblemConstants>(create_problem_constants(fmi_data));
    auto fs = std::make_unique<FullSweep>(create_fullsweep_layout(fmi_data, *pc), *pc, fmi_data);
    auto bs = std::make_unique<BoundarySweep>(create_boundarysweep_layout(fmi_data, *pc), *pc, fmi_data);

    fs->print_jacobian_sparsity_pattern();
    bs->print_jacobian_sparsity_pattern();

    return GDOP::Problem(std::move(fs), std::move(bs), std::move(pc), nullptr);
}

Problem::Problem(FMIData& fmi_data)
    : GDOP::Problem(create_gdop_problem(fmi_data))
{}

void main_fmi(FMISettings& settings) {
    Log::info("Hi from the FMI-LS-DAE interface.");
    Log::info("Path {}, Name {}", settings.path, settings.modelname);

    auto fmi_data = FMIData(settings);
    fmi_data.initialize(0.0, 0.0);
    auto problem = Problem(fmi_data);

    auto strategies = std::make_unique<GDOP::Strategies>(GDOP::Strategies::default_strategies());
    strategies->emitter = std::make_shared<GDOP::CSVEmitter>("optimal_solution.csv", false);
    strategies->mesh_refinement = std::make_shared<GDOP::L2BoundaryNorm>(settings.l2bn_p1_it, settings.l2bn_p2_it, settings.l2bn_p2_lvl);
    strategies->scaling_factory = std::make_shared<FMI::NominalScalingFactory>(fmi_data);

    auto gdop = GDOP::GDOP(problem);

    auto nlp_solver_settings = NLP::NLPSolverSettings(0, nullptr);
    nlp_solver_settings.set(NLP::Option::Hessian, NLP::HessianOption::LBFGS);
    nlp_solver_settings.set(NLP::Option::Jacobian, NLP::JacobianOption::Exact);
    nlp_solver_settings.set(NLP::Option::Gradient, NLP::GradientOption::Exact);
    // nlp_solver_settings.set(NLP::Option::IpoptDerivativeTest, true);
    nlp_solver_settings.set(NLP::Option::Tolerance, settings.tolerance);
    nlp_solver_settings.print();

    IpoptSolver::IpoptSolver ipopt_solver(gdop, nlp_solver_settings);

    auto orchestrator = GDOP::MeshRefinementOrchestrator(gdop, std::move(strategies), ipopt_solver);

    orchestrator.optimize();

    TimingTree::instance().print_tree_table();
}

} // namespace FMI
