# Consistent Initialization With `Init`

`Init` solves consistent initialization problems as static nonlinear programs.
The user defines an `Init::Problem` class, fills its `ProblemFormulation`
member with dimensions, bounds, nominals, initial values, and sparsity
patterns, and implements the required callbacks. `Init::Init` adapts this
problem to the generic `NLP::NLP` interface, so it can be solved with Ipopt or
another MOO NLP solver.

## Mathematical Form

Given an initial point `(y0, p0)`, the NLP variables are ordered as `[y, dp]`.
In the mathematical notation below, `dp` is the parameter correction. The
effective parameters passed to all callbacks are always:

```math
p = p_0 + \Delta p
```

The optimization problem is:

```math
\begin{aligned}
\min_{y, \Delta p}\quad
& J(y, p) \\
\text{s.t.}\quad
& F(y, p) = 0, \\
& g_{\mathrm{lb}} \le G(y, p) \le g_{\mathrm{ub}}, \\
& y_{\mathrm{lb}} \le y \le y_{\mathrm{ub}}, \\
& \Delta p_{\mathrm{lb}} \le \Delta p \le \Delta p_{\mathrm{ub}}, \\
& p_{\mathrm{lb}} \le p \le p_{\mathrm{ub}}.
\end{aligned}
```

`F` contains the hard initialization equations. `G` contains optional bounded
constraints, for example design-output constraints or admissible operating
ranges. User callbacks always receive `p`, never the correction variable `dp`;
`dp` is the internal NLP representation of the parameter movement shown above
as the mathematical correction variable.

## Files

The public interface and NLP adapter are implemented in:

- `src/nlp/instances/init/problem.h`: user problem definition and callbacks
- `src/nlp/instances/init/problem.cpp`: default objective implementations
- `src/nlp/instances/init/init.h`: `NLP::NLP` adapter and result access
- `src/nlp/instances/init/init.cpp`: NLP assembly and solver callback mapping

## Problem API

The main user callbacks are:

```cpp
eval_objective(y, p, obj)
eval_f(y, p, f)
eval_g(y, p, g)
eval_grad_objective(y, p, grad_y, grad_p)
eval_jacobian_f(y, p, jac_f_values)
eval_jacobian_g(y, p, jac_g_values)
eval_hessian_constraints(y, p, lambda_f, lambda_g, hes_values)
eval_hessian_objective(y, p, obj_factor, hes_values)
```

The Jacobian and Hessian are with respect to `[y, p]`. Since `p` is computed
from `p0 + dp`, the NLP maps these derivatives directly to `[y, dp]`.

## Minimal Usage

```cpp
#include <nlp/instances/init/init.h>
#include <nlp/solvers/ipopt/solver.h>
#include <nlp/solvers/nlp_solver_settings.h>

class MyInitProblem : public Init::Problem {
public:
    MyInitProblem()
    {
        formulation.y_size = 1;
        formulation.p_size = 1;
        formulation.f_size = 1;
        formulation.g_size = 0;
        formulation.objective = Init::Objective::LEAST_SQUARE_DEVIATION;

        formulation.y0 = FixedVector<f64>{0.0};
        formulation.p0 = FixedVector<f64>{1.0};
        formulation.dp0 = FixedVector<f64>{0.0};
        formulation.p_nominal = FixedVector<f64>{1.0};

        formulation.jac_f_rows = FixedVector<int>{0, 0};
        formulation.jac_f_cols = FixedVector<int>{0, 1};
    }

    void eval_f(const f64* y, const f64* p, f64* f) override
    {
        f[0] = y[0] * y[0] + p[0] - 4.0;
    }

    void eval_jacobian_f(const f64* y, const f64*, f64* jac) override
    {
        jac[0] = 2.0 * y[0];
        jac[1] = 1.0;
    }
};

MyInitProblem problem;
Init::Init init_nlp(problem);
IpoptSolver::IpoptSolver solver(init_nlp, settings);
solver.optimize();

const Init::Result& result = init_nlp.get_result();
```

If exact Hessians are not provided, configure the NLP solver with a limited
memory Hessian approximation.

## Predefined Objectives

`ProblemFormulation::objective` selects the objective implementation:

- `Objective::ZERO`: `J = 0`, with zero gradient and Hessian.
- `Objective::LEAST_SQUARE_DEVIATION`:
  ```math
  J = \sum_k \left(\frac{\Delta p_k}{p_{\mathrm{nominal},k}}\right)^2
  ```
- `Objective::USER`: the derived problem must override objective, gradient,
  and objective Hessian callbacks.

The least-squares objective uses `p_nominal`; missing, zero, or non-finite
nominals default to `1`.

## Bounds And Scaling

`Init::Init` enforces parameter bounds by intersecting direct `dp_bounds` with
shifted effective parameter bounds:

```math
\Delta p_{\mathrm{lb}} \le \Delta p \le \Delta p_{\mathrm{ub}},
\qquad
p_{\mathrm{lb}} - p_0 \le \Delta p \le p_{\mathrm{ub}} - p_0
```

By default, `Init::Init` uses `NLP::NoScaling`; the user does not need to set
any nominal data for small or already well-scaled problems.

If at least one nominal vector is provided, `Init::Init` switches to
`NLP::NominalScaling`. Nominal scaling is generated from:

- variables: `[y_nominal, dp_nominal]`, falling back to `p_nominal` for `dp`;
- constraints: `[f_nominal, g_nominal]`;
- objective: `obj_nominal`, falling back to `1`.

Missing bounds are treated as unbounded. Missing initial values default to `0`.

## Result Diagnostics

`Init::Result` stores:

- `objective`
- optimal `y`, `dp`, and effective `p`
- combined constraints `[F, G]`
- solver duals and bound multipliers
- `f_l2_norm`
- `f_max_error`
- `g_max_violation`

## Hessians

Ipopt asks for the Hessian of the Lagrangian:

```math
\sigma\nabla^2 J + \sum_i \lambda_i \nabla^2 C_i
```

where `C = [F, G]`. `ProblemFormulation::hes_rows` and `hes_cols` define the
global sparse Hessian value order with respect to `[y, p]`, which maps directly
to the NLP variables `[y, dp]`.

`Init::Init::eval_hes()` clears the output buffer, then merges contributions in
this order:

1. `Problem::eval_hessian_constraints(...)` writes the weighted constraint
   Hessian into the global sparse value buffer.
2. predefined objectives add their Hessian using slots cached when `Init::Init`
   is constructed.
3. `Objective::USER` calls `Problem::eval_hessian_objective(...)`, so custom
   objectives can use their own cached mapping and write into the same global
   sparse value buffer.

For `Objective::LEAST_SQUARE_DEVIATION`, the nonzero objective Hessian entries
are only the parameter diagonal terms. `Init::Init` computes the matching
indices in `hes_rows/hes_cols` once during construction, so no Hessian sparsity
search occurs during Ipopt iterations. If exact Hessians are used, include those
parameter diagonal entries in `hes_rows/hes_cols`; if using LBFGS, the Hessian
pattern can stay empty.
