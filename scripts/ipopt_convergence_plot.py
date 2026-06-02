import matplotlib.pyplot as plt
import numpy as np
import re
import sys

# tiny script that accepts an Ipopt output file and plots all
# quantities of the NLP w.r.t. NLP iteration

if len(sys.argv) != 2:
    print("Usage: python3 nlp_history.py <output_file>")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    log = f.read()

pattern = re.compile(
    r'^\s*(\d+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\-\d.]+)\s+([\d.e+\-]+)',
    re.MULTILINE
)

rows = pattern.findall(log)
if not rows:
    print("No IPOPT iteration data found in file.")
    sys.exit(1)

data = np.array([[float(v) for v in r] for r in rows])

iters  = data[:, 0]
obj    = data[:, 1]
inf_pr = data[:, 2]
inf_du = data[:, 3]
lg_mu  = data[:, 4]
norm_d = data[:, 5]

unscaled_cv   = re.search(r'Constraint violation\s*\.*:\s*[\d.e+\-]+\s+([\d.e+\-]+)', log)
unscaled_dual = re.search(r'Dual infeasibility\s*\.*:\s*[\d.e+\-]+\s+([\d.e+\-]+)', log)
unscaled_cv   = float(unscaled_cv.group(1))   if unscaled_cv   else None
unscaled_dual = float(unscaled_dual.group(1)) if unscaled_dual else None

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"IPOPT Convergence History — {sys.argv[1]}", fontsize=13, fontweight='bold')

axes[0, 0].semilogy(iters, obj, 'o-', color='steelblue', linewidth=1.5, markersize=4)
axes[0, 0].set_title("Objective")
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].grid(True, which='both', alpha=0.3)

axes[0, 1].semilogy(iters, inf_pr, 'o-', color='crimson', linewidth=1.5, markersize=4, label='scaled')
axes[0, 1].set_title("Primal Infeasibility — scaled space")
axes[0, 1].set_xlabel("Iteration")
axes[0, 1].grid(True, which='both', alpha=0.3)
if unscaled_cv is not None:
    axes[0, 1].axhline(unscaled_cv, color='crimson', linestyle='--', linewidth=1.2, alpha=0.7, label=f'unscaled final: {unscaled_cv:.2e}')
    axes[0, 1].legend(fontsize=8)

axes[1, 0].semilogy(iters, inf_du, 'o-', color='darkorange', linewidth=1.5, markersize=4, label='scaled')
axes[1, 0].set_title("Dual Infeasibility — scaled space")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].grid(True, which='both', alpha=0.3)
if unscaled_dual is not None:
    axes[1, 0].axhline(unscaled_dual, color='darkorange', linestyle='--', linewidth=1.2, alpha=0.7, label=f'unscaled final: {unscaled_dual:.2e}')
    axes[1, 0].legend(fontsize=8)

ax_mu = axes[1, 1]
ax_mu.plot(iters, lg_mu, 'o-', color='purple', linewidth=1.5, markersize=4, label='lg(μ)')
ax_mu.set_title("Barrier Parameter lg(μ) and Step Size ||d||")
ax_mu.set_xlabel("Iteration")
ax_mu.set_ylabel("lg(μ)", color='purple')
ax_mu.tick_params(axis='y', labelcolor='purple')
ax_mu.grid(True, alpha=0.3)

ax_d = ax_mu.twinx()
ax_d.semilogy(iters, norm_d, 's--', color='teal', linewidth=1.2, markersize=4, alpha=0.8, label='||d||')
ax_d.set_ylabel("||d||", color='teal')
ax_d.tick_params(axis='y', labelcolor='teal')

lines1, labels1 = ax_mu.get_legend_handles_labels()
lines2, labels2 = ax_d.get_legend_handles_labels()
ax_mu.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

plt.tight_layout()

out_name = sys.argv[1].rsplit('.', 1)[0] + "_convergence.png"
plt.savefig(out_name, dpi=150, bbox_inches='tight')
print(f"Saved: {out_name}")
plt.show()
