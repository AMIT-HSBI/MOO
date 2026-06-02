# General Dynamic Optimization Problem (`GDOP`)

This document explains how to describe a general dynamic optimization problem
with the `GDOP` problem class. It is written for users who want to connect
their own model source, code generator, language binding, or frontend to MOO.

Your implementation has three responsibilities:

- describe the problem dimensions, bounds, fixed values, mesh, and sparsity;
- implement callbacks for values, Jacobians, and optionally Hessians;
- choose how the optimization workflow is run, either directly or through
  pluggable strategies.

MOO also provides a C interface and an FMI interface. They follow the same
problem structure described here and can be used as examples of how to build a
custom frontend.

## Mathematical Form

A `GDOP` represents a continuous optimization problem of the form:

```math
\begin{aligned}
\min_{x(\cdot), u(\cdot), p, t_0, t_f}\quad
& M(x_0, u_0, x_f, u_f, p, t_0, t_f)
  + \int_{t_0}^{t_f} L(x(t), u(t), p, t)\,dt \\
\text{s.t.}\quad
& \dot{x}(t) = f(x(t), u(t), p, t), \\
& g_{\mathrm{lb}} \le g(x(t), u(t), p, t) \le g_{\mathrm{ub}}, \\
& r_{\mathrm{lb}} \le r(x_0, u_0, x_f, u_f, p, t_0, t_f) \le r_{\mathrm{ub}}, \\
& x_{\mathrm{lb}} \le x(t) \le x_{\mathrm{ub}}, \\
& u_{\mathrm{lb}} \le u(t) \le u_{\mathrm{ub}}, \\
& p_{\mathrm{lb}} \le p \le p_{\mathrm{ub}}, \\
& t_{0,\mathrm{lb}} \le t_0 \le t_{0,\mathrm{ub}}, \\
& t_{f,\mathrm{lb}} \le t_f \le t_{f,\mathrm{ub}}.
\end{aligned}
```

`M` is the Mayer objective, `L` is the Lagrange integrand, `f` are the dynamic
equations, `g` are path constraints, and `r` are boundary constraints. Fixed
initial or final values are provided through endpoint fixed-value vectors.

## Transcription

A `GDOP` is transcribed to a finite-dimensional NLP by direct collocation. MOO
uses flipped Legendre-Gauss-Radau collocation, where the continuous state and
control trajectories are represented at each collocation node, while static
parameters and optional free times are represented once globally. The resulting
NLP contains the collocated dynamics, path constraints, boundary constraints,
variable bounds and the discretized Mayer/Lagrange objective.

You provide the continuous problem as a `GDOP::Problem`. It contains the
dimensions, bounds, mesh, sparsity layouts, and callback objects for objective
terms, dynamics, path constraints, boundary constraints, and derivatives. MOO
then assembles the sparse NLP structure, applies scaling, maps solver variables
back to trajectories and exposes the final primal-dual trajectory, i.e. primal
optimal solution / states, controls and parameters as well as continuous duals
/ costate estimations after optimization.

## Files

The main public API is in:

- `src/nlp/instances/gdop/problem.h`: problem constants, layouts, callback base
  classes, optional simulation dynamics, and `GDOP::Problem`
- `src/nlp/instances/gdop/gdop.h`: the `NLP::NLP` implementation
- `src/nlp/instances/gdop/strategies.h`: pluggable workflow strategies
- `src/nlp/instances/gdop/orchestrator.h`: optimization workflow orchestration

## Problem Construction

To create a `GDOP::Problem`, provide:

- `GDOP::ProblemConstants`
- one `GDOP::FullSweep`
- one `GDOP::BoundarySweep`
- optionally one `GDOP::Dynamics` for simulation-based strategies

`ProblemConstants` contains the static problem description:

- sizes: `x_size`, `u_size`, `p_size`
- objective flags: `has_mayer`, `has_lagrange`
- constraint sizes: `f_size`, `g_size`, `r_size`
- bounds: `x_bounds`, `u_bounds`, `p_bounds`, `T_bounds`
- fixed endpoint values: `xu0_fixed`, `xuf_fixed`, `T_fixed`
- constraint bounds: `g_bounds`, `r_bounds`
- the initial `Mesh`

`xu` denotes one nodal pointwise block `[x, u]`. Static parameters `p` are
passed separately. If `t0` or `tf` are free, they are represented as NLP
variables and MOO updates the physical mesh during solver iterations.

## Full Sweep

`GDOP::FullSweep` evaluates pointwise functions at every collocation node:

```math
L(x, u, p, t),\qquad
f(x, u, p, t),\qquad
g(x, u, p, t)
```

Derive from `GDOP::FullSweep` and implement:

```cpp
void callback_eval(const f64* xu_nlp, const f64* p) override;
void callback_jac(const f64* xu_nlp, const f64* p) override;
void callback_hes(const f64* xu_nlp,
                  const f64* p,
                  const FixedField<f64, 2>& lagrange_factors,
                  const f64* lambda) override;
```

Inside the callbacks, iterate over all mesh intervals and nodes:

```cpp
for (int i = 0; i < pc.mesh->intervals; i++) {
    for (int j = 0; j < pc.mesh->nodes[i]; j++) {
        const f64* xu = get_xu_ij(xu_nlp, i, j);
        f64 t = pc.mesh->t[i][j];
        f64* values = get_eval_buffer(i, j);
    }
}
```

Useful helpers are:

- `get_xu_ij(xu_nlp, i, j)`: node-local `[x, u]`
- `get_lambda_ij(lambda, i, j)`: node-local multipliers for `[f, g]`
- `get_eval_buffer(i, j)`: pointwise value buffer
- `get_jac_buffer(i, j)`: pointwise Jacobian buffer
- `get_hes_buffer(i, j)`: pointwise Hessian buffer except pure `p,p`
- `get_pp_hes_buffer()`: shared pure `p,p` Hessian buffer

## Boundary Sweep

`GDOP::BoundarySweep` evaluates endpoint functions:

```math
M(x_0, u_0, x_f, u_f, p, t_0, t_f),\qquad
r(x_0, u_0, x_f, u_f, p, t_0, t_f)
```

Derive from `GDOP::BoundarySweep` and implement:

```cpp
void callback_eval(const f64* xu0,
                   const f64* xuf,
                   const f64* p,
                   f64 t0,
                   f64 tf) override;

void callback_jac(const f64* xu0,
                  const f64* xuf,
                  const f64* p,
                  f64 t0,
                  f64 tf) override;

void callback_hes(const f64* xu0,
                  const f64* xuf,
                  const f64* p,
                  f64 t0,
                  f64 tf,
                  f64 mayer_factor,
                  const f64* lambda) override;
```

Write endpoint values and derivatives to `get_eval_buffer()`,
`get_jac_buffer()`, and `get_hes_buffer()`.

## Layouts And Buffers

Before constructing the problem, define the layouts:

- `FullSweepLayout`: optional `L`, `f`, `g`, full-sweep Jacobian sparsity,
  full-sweep Hessian sparsity, and pure parameter Hessian sparsity
- `BoundarySweepLayout`: optional `M`, `r`, boundary Jacobian sparsity, and
  boundary Hessian sparsity

Each evaluation, Jacobian, and Hessian entry has a `buf_index`. This is the
contract between your callback and MOO:

- you may provide derivative entries in any order that is convenient for your
  frontend;
- MOO uses the sparsity metadata and `buf_index` values to reshuffle your
  buffers into its internal sparse NLP representation;
- the callback must write each value to the buffer position declared by its
  `buf_index`.

This means the user-facing derivative layout can look like COO, CSC, CSR, a
generated code layout, or any other flat sparse layout, as long as the sparsity
metadata and callback buffer writes agree.

## Derivatives

`callback_jac(...)` writes the Jacobian values declared in the corresponding
layout. The row, column, and `buf_index` metadata define which derivative each
buffer entry represents.

Hessian sparsity is lower triangular: every declared Hessian entry must satisfy
`row >= col`. The Hessian callbacks receive already transformed multipliers:

- `FullSweep::callback_hes(...)` receives node-local multipliers for `[f, g]`
  and `lagrange_factors[i][j]` for `L`;
- `BoundarySweep::callback_hes(...)` receives multipliers for `r` and
  `mayer_factor` for `M`.

The callback writes the weighted Hessian of the local Lagrangian. For the full
sweep this is:

```math
\ell_{ij}\nabla^2 L
+ \lambda_f^\mathsf{T}\nabla^2 f
+ \lambda_g^\mathsf{T}\nabla^2 g
```

For the boundary sweep this is:

```math
\mu_M\nabla^2 M + \lambda_r^\mathsf{T}\nabla^2 r
```

### Pure Parameter Hessians

Pure full-sweep `p,p` Hessian entries are special. Parameters are global, but
`L`, `f`, and `g` are evaluated at every collocation node. Therefore all
full-sweep pure parameter Hessian contributions share one buffer:

```cpp
f64* pp_hes = get_pp_hes_buffer();
```

This buffer is shared by all node evaluations in one Hessian callback. You
must always accumulate into it with `+=`:

```cpp
pp_hes[buf_index] += node_contribution;
```

Do not assign with `=`. MOO zeroes the shared `pp` buffer before the full-sweep
Hessian callback, then inserts the accumulated values into the global NLP
Hessian.

All other full-sweep Hessian entries are node-local and should be written to
`get_hes_buffer(i, j)` for the current node. Boundary Hessian entries are
written to the boundary Hessian buffer.

If exact Hessians are not available, configure the NLP solver to use a limited
memory Hessian approximation.

## Optional Dynamics

`GDOP::Dynamics` is not required for the NLP transcription. Provide it if you
want to use simulation-based strategies, such as simulation initialization or
simulation-based verification.

Implement:

```cpp
void eval(const f64* x, const f64* u, const f64* p, f64 t, f64* f, void* user_data);
void jac(const f64* x, const f64* u, const f64* p, f64 t, f64* dfdx, void* user_data);
```

## Solving

A minimal solve constructs the problem, wraps it as a `GDOP::GDOP` NLP
instance, creates solver settings, and runs the solver:

```cpp
GDOP::Problem problem = create_my_problem();
GDOP::GDOP gdop(problem);

NLP::NLPSolverSettings settings(argc, argv);
IpoptSolver::IpoptSolver solver(gdop, settings);

gdop.set_initial_guess(create_initial_guess());
solver.optimize();

const PrimalDualTrajectory* solution = gdop.get_optimal_solution();
```

For workflows with initialization, scaling, mesh refinement, verification, and
output, use the orchestrator:

```cpp
auto strategies = std::make_unique<GDOP::Strategies>(
    GDOP::Strategies::default_strategies());

strategies->initialization = std::make_shared<MyInitialization>();
strategies->mesh_refinement = std::make_shared<MyMeshRefinement>();
strategies->emitter = std::make_shared<MyEmitter>();

GDOP::MeshRefinementOrchestrator orchestrator(gdop, std::move(strategies), solver);
orchestrator.optimize();
```

After optimization, `gdop.get_optimal_solution()` returns the reconstructed
`PrimalDualTrajectory`, including primals, constraint costates, and lower/upper
bound multipliers.

## Strategies

Strategies customize the optimization workflow without changing the problem
callbacks. You can use the defaults or provide your own implementations.

The strategy interfaces are:

- `Initialization`: creates the first `PrimalDualTrajectory`
- `RefinedInitialization`: maps a solution to a refined mesh
- `Simulation`: simulates a full horizon
- `SimulationStep`: simulates one segment
- `MeshRefinement`: decides whether and how to refine the mesh
- `Interpolation`: interpolates values between meshes
- `Emitter`: writes or prints the final trajectory
- `Verifier`: checks the final solution
- `ScalingFactory`: creates NLP scaling

`MeshRefinementOrchestrator` uses the strategies in this order:

1. reset strategies
2. create the initial guess
3. set scaling on the `GDOP::GDOP` NLP instance
4. solve the current NLP
5. ask the mesh-refinement strategy for a `MeshUpdate`
6. interpolate the solution to the refined mesh
7. update the `GDOP::GDOP` NLP instance with the refined mesh
8. repeat until no mesh update is requested
9. verify and emit the final solution

Existing strategies include constant initialization, simulation-based
initialization, Radau-based simulation, polynomial interpolation, CSV/print
emitters, simulation verification, no-scaling, and L2-boundary-norm mesh
refinement. Adaptive mesh refinement for this workflow is discussed in
"Enhancing Collocation-Based Dynamic Optimization through Adaptive Mesh
Refinement", Linus Langenkamp and Bernhard Bachmann, 16th International
Modelica & FMI Conference, 2025, DOI: 10.3384/ecp218127.

## C Interface

MOO also exposes a C interface for users who do not want to derive C++ classes
directly. The C interface follows the same structure: you provide dimensions,
bounds, sparsity layouts, callback function pointers, and optional workflow
settings.

The callback signatures mirror the sweep model:

```c
eval_lfg(xu, p, t, data, out, user_data)
jac_lfg(xu, p, t, data, out, user_data)
hes_lfg(xu, p, lambda, obj_factor, t, data, out, out_pp, user_data)

eval_mr(xu0, xuf, p, t0, tf, data_t0, data_tf, out, user_data)
jac_mr(xu0, xuf, p, t0, tf, data_t0, data_tf, out, user_data)
hes_mr(xu0, xuf, p, lambda, obj_factor, t0, tf, data_t0, data_tf, out, user_data)
```

The same `buf_index` rule applies. `out` and `out_pp` are flat buffers whose
meaning is defined by the sparsity metadata supplied with the problem.

For `hes_lfg`, `out_pp` is the shared pure parameter Hessian buffer. If the
callback is invoked node by node, accumulate into `out_pp` with `+=`.

This interface is work-in-progress and mainly designed as a code generation target. It
aims to be general, but is intentionally restricted to the bare minimum.

## FMI Interface

The FMI interface builds a `GDOP::Problem` from FMU value references. Users select
objective outputs, tunable parameters, controls, path constraints, boundary
constraints, fixed start values, and nominals. This is very experimental and
will be extended once the layered standard LS-DAE comes live.
