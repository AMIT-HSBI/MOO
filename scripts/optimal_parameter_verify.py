### Find optimal parameter p for the MWE
###
###     min_{u(t), p} int_{0}^{100} 1/2 * (x² + u²) dt
###
### s.t.        x'(t)  = -x(t) + u(t) + p
###             x(0)   = 1.5
###             x(100) = 1.0
###
### Verifies the optimal parameter: user_p = 0.02500176076771822 found by MOO
###

# Golden-section search around 0.025 ± 1e-3 with high tolerance (1e-12).
import numpy as np
from math import sqrt, exp
from scipy.integrate import simpson


def optimal_profiles(p, T=100.0, N=20000):
    # reduced N to speed up but still accurate (exponentials dominate)
    s = sqrt(2.0)
    E = exp(s * T)
    A = (-(1.5 - p / 2.0) + (1.0 - p / 2.0) * E) / (E**2 - 1.0)
    B = (E * ((1.5 - p / 2.0) * E - (1.0 - p / 2.0))) / (E**2 - 1.0)
    t = np.linspace(0.0, T, N)
    x = p / 2.0 + A * np.exp(s * t) + B * np.exp(-s * t)
    u = A * (1.0 + s) * np.exp(s * t) + B * (1.0 - s) * np.exp(-s * t) - p / 2.0
    J = simpson(x**2 + u**2)
    return A, B, J


def J_of_p(p):
    return optimal_profiles(p)[2]


def argmin_golden(f, pL, pU, tol=1e-12, maxit=200):
    gr = (np.sqrt(5) - 1) / 2.0
    c = pU - gr * (pU - pL)
    d = pL + gr * (pU - pL)
    fc, fd = f(c), f(d)
    it = 0
    while abs(pU - pL) > tol and it < maxit:
        if fc > fd:
            pL = c
            c = d
            fc = fd
            d = pL + gr * (pU - pL)
            fd = f(d)
        else:
            pU = d
            d = c
            fd = fc
            c = pU - gr * (pU - pL)
            fc = f(c)
        it += 1
    p_star = (pL + pU) / 2.0
    return float(p_star), float(f(p_star)), it


pL = 0.025 - 1e-3
pU = 0.025 + 1e-3
p_star, J_star, iters = argmin_golden(J_of_p, pL, pU, tol=1e-14, maxit=1000)

A, B, J_final = optimal_profiles(p_star)

print("search interval: [{:.15f}, {:.15f}]".format(pL, pU))
print("iterations:", iters)
print("p* = {:.17f}".format(p_star))
print("J(p*) = {:.12f}".format(J_star))
print("A = {:.12e}".format(A))
print("B = {:.12e}".format(B))

moo_p = 0.02500176076771822
diff = p_star - moo_p
print("difference to moo_p: {:.17e}".format(diff))
