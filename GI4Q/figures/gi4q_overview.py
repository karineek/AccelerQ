"""GI4Q overview figure: AccelerQ pipeline with the GI4Q Phase-3 search loop expanded.
Outputs gi4q_overview.pdf and gi4q_overview.png next to this script."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1f2933"
MUTED = "#6b7280"
GREY_FILL = "#eef0f3"
GREY_EDGE = "#9aa3ad"
ACC_FILL = "#dbeafe"
ACC_EDGE = "#1d4ed8"
ACC_TXT = "#1e3a8a"

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 110)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, title, body="", changed=False, title_size=9.5, body_size=8):
    fc, ec, tc = (ACC_FILL, ACC_EDGE, ACC_TXT) if changed else (GREY_FILL, GREY_EDGE, INK)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6 if changed else 1.0, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=tc, zorder=3)
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=body_size,
                color=INK, linespacing=1.35, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=tc, zorder=3)


def arrow(x1, y1, x2, y2, color=INK, style="-|>", lw=1.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ---------------- Top: AccelerQ pipeline ----------------
ax.text(1, 60.5, "AccelerQ pipeline", fontsize=11, fontweight="bold", color=INK, va="top")
ty, th, tw = 44.5, 12.5, 22.5
xs = [1.5, 29.0, 56.5, 84.5]
box(xs[0], ty, tw, th, "Phase 1 · Data",
    "Run QE implementation on\nsmall systems (4–8 qubits)\nwith random hyperparameters")
box(xs[1], ty, tw, th, "Phase 2 · Surrogate",
    "Train XGBoost on\n(hyperparameters, Hamiltonian)\n→ predicted energy")
box(xs[2], ty, tw, th, "Phase 3 · Search (GI4Q)", title_size=9.0, body=
    "GA over hyperparameters on\n20–24-qubit Hamiltonians,\nfitness = surrogate prediction", changed=True)
box(xs[3], ty, 24.0, th, "Stage B · Validation",
    "Run QE implementation with\nbest vector on MPS simulator\n→ Task Score (energy)")
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + tw + 0.6, ty + th / 2, b - 0.6, ty + th / 2)

# Zoom bracket from Phase 3 to the loop panel
panel_top = 37.5
ax.plot([xs[2] + 0.5, 3.0], [ty - 0.6, panel_top + 0.4], color=ACC_EDGE, lw=0.8, ls=(0, (3, 2)))
ax.plot([xs[2] + tw - 0.5, 107.0], [ty - 0.6, panel_top + 0.4], color=ACC_EDGE, lw=0.8, ls=(0, (3, 2)))

# ---------------- Bottom: GI4Q search loop ----------------
ax.add_patch(Rectangle((1.5, 1.0), 107.0, panel_top - 1.0, fc="white", ec=ACC_EDGE, lw=0.9, zorder=0))
ax.text(3.5, panel_top - 1.4, "GI4Q Phase-3 search loop  (49 iterations, 4 offspring / iteration)",
        fontsize=10, fontweight="bold", color=ACC_TXT, va="top")

ly, lh = 15.5, 16.5
box(3.5, ly, 15.5, lh, "Initialise", "499 random seeds\nfiltered by static and\nsemi-dynamic tests,\nscored by surrogate")
box(22.5, ly, 18.0, lh, "Selection", "Tournament, k = 3,\nover the top-20 pool\n\n(was: uniform pick\nfrom top-20)", changed=True)
box(44.0, ly, 18.0, lh, "Crossover", "Baseline 5 operators\n+ BLX-α (α = 0.5)\nas a 6th operator\n\n(was: 5 operators)", changed=True)
box(65.5, ly, 18.5, lh, "Mutation", "Gaussian, p = 0.5,\nσ: 10% → 1% linear,\nscaled per gene\n\n(was: none)", changed=True)
box(87.5, ly, 19.0, lh, "Evaluate", "Validity tests, then\nXGBoost prediction;\nadd offspring\nto corpus")

for x1, x2 in [(19.0, 22.5), (40.5, 44.0), (62.0, 65.5), (84.0, 87.5)]:
    arrow(x1 + 0.6, ly + lh / 2, x2 - 0.6, ly + lh / 2)

# Maintenance row: Evaluate -> (every 5th) prune+noise -> elitism -> back to Selection
my, mh = 2.6, 10.0
box(22.5, my, 18.0, mh, "Elitism", "Reinstate best-so-far\nafter pruning",
    changed=True, title_size=8.8, body_size=7.8)
box(44.0, my, 46.0, mh, "Every 5th iteration (unchanged)",
    "Prune corpus to best 50%, inject fresh random noise", title_size=8.8, body_size=8)
box(94.0, my, 12.5, mh, "Output", "Best after 49\niters → Stage B", title_size=8.8, body_size=7.8)


def path(pts, color=INK):
    xs_, ys_ = zip(*pts[:-1])
    ax.plot(list(xs_) + [pts[-1][0]], list(ys_) + [pts[-1][1]], color=color, lw=1.2, zorder=1,
            solid_capstyle="butt")
    arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color)


path([(92.0, ly - 0.6), (92.0, my + mh / 2), (90.6, my + mh / 2)])   # evaluate -> prune
path([(43.4, my + mh / 2), (41.1, my + mh / 2)])                       # prune -> elitism
path([(31.5, my + mh + 0.6), (31.5, ly - 0.6)])                        # elitism -> selection
path([(100.25, ly - 0.6), (100.25, my + mh + 0.6)])                    # evaluate -> output
ax.text(32.3, (ly + my + mh) / 2, "next iteration", fontsize=7, color=MUTED, va="center")

# Legend
ax.add_patch(FancyBboxPatch((70.0, 59.0), 2.2, 1.6, boxstyle="round,pad=0.2", fc=ACC_FILL, ec=ACC_EDGE, lw=1.4))
ax.text(73.0, 59.8, "Changed in GI4Q", fontsize=8, va="center", color=INK)
ax.add_patch(FancyBboxPatch((88.0, 59.0), 2.2, 1.6, boxstyle="round,pad=0.2", fc=GREY_FILL, ec=GREY_EDGE, lw=1.0))
ax.text(91.0, 59.8, "Unchanged from AccelerQ", fontsize=8, va="center", color=INK)

out = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out, "gi4q_overview.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out, "gi4q_overview.png"), dpi=300, bbox_inches="tight")
print("saved")
