#ifndef OPT_OM_INFO_GDOP_H
#define OPT_OM_INFO_GDOP_H

#include "simulation_data.h"
#include "simulation/solver/gbode_main.h"
#include "simulation/solver/external_input.h"

#include "sim_runtime_ext.h"

#include <base/nlp_structs.h>
#include <base/fixed_vector.h>
#include <nlp/solvers/nlp_solver_settings.h>

#include <memory>

namespace OpenModelica {

/* foward decl */
struct ExchangeJacobians;
struct ExchangeHessians;

struct InfoGDOP {
    DATA* data;               // pointer to OM data object
    threadData_t* threadData; // pointer to OM threadData object
    int argc;                 // command-line arg count OM
    char** argv;              // command-line args OM

    /* problem sizes */
    int x_size;
    int u_size;
    int p_size;
    int xu_size;
    int f_size;
    int g_size;
    int r_size;

    /* objective structure */
    bool mayer_exists;
    bool lagrange_exists;

    /* addresses in realVars */
    modelica_real* address_mayer_real_vars;
    modelica_real* address_lagrange_real_vars;

    /* realVars variables indices */
    const int index_x_real_vars = 0;
    int index_der_x_real_vars;
    int index_mayer_real_vars = -1;
    int index_lagrange_real_vars = -1;
    FixedVector<int> u_indices_real_vars;
    int index_g_real_vars;
    int index_r_real_vars;

    /* exchange format for Jacobians OM <-> OPT */
    std::unique_ptr<ExchangeJacobians> exc_jac;

    /* numerical Hessians */
    std::unique_ptr<ExchangeHessians> exc_hes;

    /* time horizon */
    f64 model_start_time; // model start_time
    f64 model_stop_time;  // model stop_time
    f64 tf;               // tf = start - stop, since t0 = 0 for OPT
    int intervals;        // model interval count
    int stages;           // stage count

    SOLVER_METHOD user_ode_solver = S_GBODE;

    InfoGDOP(DATA* data, threadData_t* threadData, int argc, char** argv);

    void set_omc_flags(NLP::NLPSolverSettings& nlp_solver_settings);
    void set_time_horizon(int steps);
    void set_user_solver();
};

/**
 * @brief Constructs and initializes OpenModelica Jacobians and their COO mappings for optimization.
 *
 * This constructor sets up the Jacobian matrices (A, B, C, D) using OpenModelica's internal
 * CSC (Compressed Sparse Column) structures and transforms them into COO (Coordinate) format
 * as required by the optimization backend.
 *
 * OpenModelica provides the following Jacobians in CSC format:
 *   - A = (f_{xu})^T
 *   - B = (f_{xu}, L_{xu}, g_{xu})^T
 *   - C = (f_{xu}, L_{xu}, M_{xu}, g_{xu})^T
 *   - D = (r_{xu})^T
 *
 * ### LFG Block (B matrix)
 * - The B matrix includes rows for f, L, and g in that order (`fLg`).
 * - It is converted directly from CSC to COO without reordering.
 * - Although the sparsity structure is stored in COO format, the numerical values remain in CSC order.
 * - This allows the original CSC evaluation buffer to be passed directly to the optimizer callbacks,
 *   since the buffer indices (`buf_index`) in the sparsity structure correctly point to CSC entries.
 *
 * ### MR Block (C and D matrices)
 * - **Mayer term (M)** is embedded in the C matrix:
 *   - C has row order: f, L, M, g.
 *   - It is converted to COO format, and the M row is moved to the top to match the optimizer's expected order.
 *   - During evaluation, the full C matrix is evaluated in CSC format into a temporary buffer.
 *   - Using the `coo_to_csc` mapping, only the entries of the M row (now first in COO) are extracted
 *     into the final buffer passed to the optimizer.
 *
 * - **Boundary constraints (r)** are represented in the D matrix:
 *   - D contains only `r_{xu}` and is converted directly to COO.
 *   - Since the optimizer expects `[M, r]` ordering in the MR block, a row offset is applied
 *     during evaluation to place `r` entries after the Mayer entries in the final buffer.
 *
 * Throughout this process, the `csc_to_coo` and `coo_to_csc` permutation arrays maintain consistent
 * mapping between OpenModelica’s CSC structures and the COO format expected by the optimization routines.
 */

struct ExchangeJacobian {
    JACOBIAN* jacobian;
    bool exists;
    CscToCoo sparsity;
    FixedVector<modelica_real> buffer;

    ExchangeJacobian(InfoGDOP& info,
                     int index,
                     initialAnalyticalJacobian_func_ptr initial_jacobian,
                     int move_to_first_row = -1,
                     int nnz_offset = 0) :
    jacobian(&(info.data->simulationInfo->analyticJacobians[index])),
    exists(static_cast<bool>(initial_jacobian(info.data, info.threadData, jacobian) == 0)),
    sparsity(exists ? CscToCoo::from_csc(reinterpret_cast<int*>(jacobian->sparsePattern->leadindex),
                                         reinterpret_cast<int*>(jacobian->sparsePattern->index),
                                         static_cast<int>(jacobian->sizeCols),
                                         static_cast<int>(jacobian->sparsePattern->numberOfNonZeros),
                                         move_to_first_row,
                                         nnz_offset)
                    : CscToCoo()),
    buffer(FixedVector<modelica_real>(sparsity.nnz)) {}
};

struct ExchangeJacobians {
    ExchangeJacobian A;
    ExchangeJacobian B;
    ExchangeJacobian C;
    ExchangeJacobian D;

    ExchangeJacobians(InfoGDOP& info);
};

struct ExchangeHessian {
    HESSIAN_PATTERN* hessian;
    bool exists;
    ExtrapolationData* extr;

    /* some workspace memory for output of Hessian, use if needed */
    FixedVector<modelica_real> buffer;

    /* some workspace memory for dual multiplicators, use if needed */
    FixedVector<modelica_real> lambda;

    /* wrapper args for Richardson extrapolation */
    HessianFiniteDiffArgs args;

    ExchangeHessian(InfoGDOP& info,
                    ExchangeJacobian& inducing_jacobian) :
    hessian(generateHessianPattern(inducing_jacobian.jacobian)),
    exists(inducing_jacobian.exists),
    extr(!exists ? nullptr : initExtrapolationData(hessian->lnnz, 5)),
    buffer(FixedVector<modelica_real>(!exists ? 0 : hessian->lnnz)),
    lambda(FixedVector<modelica_real>(!exists ? 0 : hessian->numFuncs)),
    args(HessianFiniteDiffArgs{info.data, info.threadData, hessian, info.u_indices_real_vars.raw(), lambda.raw(), nullptr}) {}

    ~ExchangeHessian() {
        // delete c mallocs
        if (extr) freeExtrapolationData(extr);
        if (hessian) freeHessianPattern(hessian);
    }

    // copy = delete
    ExchangeHessian(const ExchangeHessian&) = delete;
    ExchangeHessian& operator=(const ExchangeHessian&) = delete;
};

struct ExchangeHessians {
    ExchangeHessian A;
    ExchangeHessian B;
    ExchangeHessian C;
    ExchangeHessian D;

    /* mapping of OM C and D HESSIAN_PATTERN indices -> Mr buffer indices, will be set in 'init_hes_mr()' */
    FixedVector<std::pair<int, int>> C_to_Mr_buffer;
    FixedVector<std::pair<int, int>> D_to_Mr_buffer;

    ExchangeHessians(InfoGDOP& info);
};

} // namespace OpenModelica

#endif // OPT_OM_INFO_GDOP_H
