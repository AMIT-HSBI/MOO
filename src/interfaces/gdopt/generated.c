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

#include <interfaces/c/structures.h>
#include <interfaces/gdopt/main_gdopt.h>
#include <interfaces/gdopt/generated.h>

// === problem sizes (compile const) ===

#define X_SIZE 1
#define U_SIZE 1
#define P_SIZE 0
#define RP_SIZE 0

#define R_SIZE 0
#define G_SIZE 0

#define HAS_MAYER true
#define HAS_LAGRANGE true

// === declare global variables (values can be influenced by runtime parameters - _rp) ===

bounds_t globl_x_bounds[X_SIZE] = { { -DBL_MAX, DBL_MAX } };
bounds_t globl_u_bounds[U_SIZE] = { { -5.0, 5.0 } };
bounds_t globl_p_bounds[P_SIZE];

bounds_t globl_g_bounds[G_SIZE];
bounds_t globl_r_bounds[R_SIZE];

optional_value_t globl_x0_fixed[X_SIZE] = { {0.0, true} };
optional_value_t globl_xf_fixed[X_SIZE] = { {0.0, false} };

f64 globl_x_nominal[X_SIZE];
f64 globl_u_nominal[U_SIZE];
f64 globl_p_nominal[P_SIZE];

f64 globl_obj_nominal;
f64 globl_f_nominal[X_SIZE];
f64 globl_g_nominal[G_SIZE];
f64 globl_r_nominal[R_SIZE];

f64 _rp[RP_SIZE];

// === sparsity and evaluation structures (compile const) ===

eval_structure_t globl_lfg_eval = {
    .buf_index = (int[]){0, 1}
};

coo_t globl_lfg_jac = {
    .row = (int[]){0, 0, 1, 1},
    .col = (int[]){0, 1, 0, 1},
    .buf_index = (int[]){0, 1, 2, 3},
    .nnz = 4
};

coo_t globl_lfg_lt_hes = {
    .row = (int[]){0, 1},
    .col = (int[]){0, 1},
    .buf_index = (int[]){0, 1},
    .nnz = 2
};

eval_structure_t globl_mr_eval = {
    .buf_index = (int[]){0}
};

coo_t globl_mr_jac = {
    .row = (int[]){0, 0},
    .col = (int[]){1, 2},
    .buf_index = (int[]){0, 1},
    .nnz = 2
};

coo_t globl_mr_lt_hes = {
    .row = (int[]){},
    .col = (int[]){},
    .buf_index = (int[]){},
    .nnz = 0
};

void update_c_problem(void* ctx) { return; }

void eval_f(f64 t, const f64* x, const f64* u, const f64* p, const f64* w, f64* dxdt, void* user_data) {
    dxdt[0] = x[0] + u[0];
}

void jac(f64 t, const f64* x, const f64* u, const f64* p, const f64* w, f64* J, void* user_data) {
    J[0] = 1; /* f_x = 1 */
}

// [L, f, g]
void eval_lfg(const f64* xu, const f64* p, f64 t, f64* out) {
    const f64* x = xu;
    const f64* u = xu + X_SIZE;

    out[0] /* L */ = 0.5 * (x[0] * x[0] + u[0] * u[0]);
    out[1] /* f */ = x[0] + u[0];
}

// ∇ [L, f, g]
void jac_lfg(const f64* xu, const f64* p, f64 t, f64* out) {
    const f64* x = xu;
    const f64* u = xu + X_SIZE;

    out[0] = x[0]; /* L_x = x */
    out[1] = u[0]; /* L_u = u */
    out[2] = 1; /* f_x = 1 */
    out[3] = 1; /* f_u = 1 */
}

// σ ∇² L + λ^T ∇² [f, g] (lower triangle)
void hes_lfg(const f64* xu, const f64* p, const f64* lambda, const f64 obj_factor, f64 t, f64* out) {
    out[0] = obj_factor; /* ℒ_xx = 1 * sigma */
    out[1] = obj_factor; /* ℒ_uu = 1 * sigma */
}

// [M, r]
void eval_mr(const f64* x0, const f64* xuf, const f64* p, f64 t0, f64 tf, f64* out) {
    const f64* xf = xuf;
    const f64* uf = xuf + X_SIZE;

    out[0] = xf[0] + uf[0];
}

// ∇ [M, r]
void jac_mr(const f64* x0, const f64* xuf, const f64* p, f64 t0, f64 tf, f64* out) {
    out[0] = 1;
    out[1] = 1;
}

// σ ∇² M + λ^T ∇² r (lower triangle)
void hes_mr(const f64* x0, const f64* xuf, const f64* p, const f64* lambda, const f64 obj_factor, f64 t0, f64 tf, f64* out) {

}

c_callbacks_t globl_callbacks = {
    update_c_problem,
    eval_lfg,
    jac_lfg,
    hes_lfg,
    eval_mr,
    jac_mr,
    hes_mr
};

c_problem_t globl_c_problem = {
    .x_size = X_SIZE,
    .u_size = U_SIZE,
    .xu_size = X_SIZE + U_SIZE,
    .p_size = P_SIZE,
    .rp_size = RP_SIZE,
    .r_size = R_SIZE,
    .g_size = G_SIZE,
    .has_mayer = HAS_MAYER,
    .has_lagrange = HAS_LAGRANGE,
    .x_bounds = globl_x_bounds,
    .u_bounds = globl_u_bounds,
    .p_bounds = globl_p_bounds,
    .r_bounds = globl_r_bounds,
    .g_bounds = globl_g_bounds,
    .x0_fixed = globl_x0_fixed,
    .xf_fixed = globl_xf_fixed,
    .x_nominal = globl_x_nominal,
    .u_nominal = globl_u_nominal,
    .p_nominal = globl_p_nominal,
    .obj_nominal = &globl_obj_nominal,
    .f_nominal = globl_f_nominal,
    .g_nominal = globl_g_nominal,
    .r_nominal = globl_r_nominal,
    .lfg_eval  = &globl_lfg_eval,
    .lfg_jac  = &globl_lfg_jac,
    .lfg_lt_hes  = &globl_lfg_lt_hes,
    .mr_eval = &globl_mr_eval,
    .mr_jac = &globl_mr_jac,
    .mr_lt_hes = &globl_mr_lt_hes,
    .callbacks = &globl_callbacks
};


int main_generated(int argc, char** argv) {
    main_gdopt(argc, argv, &globl_c_problem);
    return 0;
}
