"""Landing-page art in the Ink & Cobalt palette (decorative, not real data).

    python tools/make_spots.py images

Writes:
  hero-cover.png            dark hero behind the headline (2400 x 1000)
  spot-intervals[-inverse].png, spot-dag[-inverse].png,
  spot-contours[-inverse].png, spot-field[-inverse].png   (800 x 800, transparent)
The -inverse versions are drawn in light colors for the solid color blocks.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Circle

OUT = sys.argv[1] if len(sys.argv) > 1 else "images"
os.makedirs(OUT, exist_ok=True)
INK, COBALT, DEEP, PAPER = "#121417", "#2F5BD3", "#1E43A8", "#FCFCFB"
RAMP = ["#EEF2FC", "#D9E2F8", "#BFCEF3", "#9FB4EC", "#7C98E4", "#5A7EDC", "#2F5BD3", "#2549B0", "#1A3687", "#0F1E3D"]


def surface(seed, nx=900, ny=375, xmax=2.4, peaks=10):
    r = np.random.default_rng(seed)
    x = np.linspace(0, xmax, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for _ in range(peaks):
        cx, cy, s = r.uniform(0, xmax), r.uniform(-0.2, 1.2), r.uniform(0.15, 0.45)
        Z += r.uniform(0.5, 1.2) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2))
    return X, Y, Z


# ---- Dark hero ---------------------------------------------------------------
fig = plt.figure(figsize=(24, 10), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
X, Y, Z = surface(11)
lv = np.linspace(Z.min(), Z.max(), 30)
cm = LinearSegmentedColormap.from_list("h", ["#2A3D6B", COBALT, "#7C98E4", "#D9E2F8"])
ax.contour(X, Y, Z, levels=lv, cmap=cm, linewidths=1.9)
# soft dark wash on the left so the headline stays readable
grad = np.linspace(1, 0, 600) ** 1.6
ax.imshow(np.tile(grad, (10, 1)), extent=[0, 1.5, 0, 1], cmap=LinearSegmentedColormap.from_list("w", [(0.07, 0.08, 0.09, 0), (0.07, 0.08, 0.09, 0.92)]),
          aspect="auto", zorder=3)
ax.set_xlim(0, 2.4); ax.set_ylim(0, 1)
fig.savefig(os.path.join(OUT, "hero-cover.png"), dpi=100, facecolor=INK)
plt.close(fig)


# ---- Spot illustrations ---------------------------------------------------------
def spot(name, draw):
    for inverse in (False, True):
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88]); ax.set_axis_off()
        fig.patch.set_alpha(0); ax.patch.set_alpha(0)
        draw(ax, inverse)
        fn = f"spot-{name}{'-inverse' if inverse else ''}.png"
        fig.savefig(os.path.join(OUT, fn), dpi=100, transparent=True)
        plt.close(fig)


def intervals(ax, inv):
    r = np.random.default_rng(3)
    n = 48
    t = np.linspace(0, 1, n)
    base = np.exp(-((t - 0.4) ** 2) / 0.02) + 0.6 * np.exp(-((t - 0.68) ** 2) / 0.03) + 0.08
    v = base * r.uniform(0.85, 1.1, n); v = v / v.max()
    bar = PAPER if inv else COBALT
    line = PAPER if inv else INK
    ax.bar(np.arange(n), v, width=0.7, color=bar, alpha=0.55 if inv else 0.85)
    ax.plot(np.arange(n), base / base.max(), color=line, lw=5)
    ax.axhline(0, color=line, lw=4)
    ax.set_xlim(-1, n); ax.set_ylim(-0.02, 1.08)


def dag(ax, inv):
    node_fill = PAPER if inv else COBALT
    node_edge = PAPER if inv else INK
    edge = PAPER if inv else INK
    pos = {"Z": (0.5, 0.82), "T": (0.18, 0.3), "Y": (0.82, 0.3), "U": (0.5, 0.08)}
    for a, b in [("Z", "T"), ("Z", "Y"), ("T", "Y"), ("U", "Y")]:
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=48, lw=6, color=edge,
                                     shrinkA=60, shrinkB=60, linestyle="--" if a == "U" else "-"))
    for k, (x, y) in pos.items():
        filled = k in ("T", "Y")
        ax.add_patch(Circle((x, y), 0.1, facecolor=node_fill if filled else "none", edgecolor=node_edge, lw=6))
        ax.text(x, y, k, ha="center", va="center", fontsize=40, fontweight="bold",
                color=(INK if inv else PAPER) if filled else node_edge, family="DejaVu Sans")
    ax.set_xlim(0, 1); ax.set_ylim(-0.08, 1)


def contours(ax, inv):
    X, Y, Z = surface(5, 400, 400, 1.0, 6)
    lv = np.linspace(Z.min(), Z.max(), 16)
    if inv:
        ax.contour(X, Y, Z, levels=lv, colors=[PAPER], linewidths=3, alpha=0.85)
    else:
        ax.contour(X, Y, Z, levels=lv, cmap=LinearSegmentedColormap.from_list("c", ["#9FB4EC", COBALT, INK]), linewidths=3)
    ax.add_patch(Circle((0.62, 0.55), 0.035, color=PAPER if inv else INK))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")


def field(ax, inv):
    gx, gy = np.meshgrid(np.arange(14), np.arange(14))
    gx, gy = gx.ravel(), gy.ravel()
    w = 0.5 + 0.5 * np.sin(gx / 2.2 + np.cos(gy / 2.0) * 1.4) * np.cos(gy / 3.0)
    s = 40 + 900 * w ** 2
    if inv:
        ax.scatter(gx, gy, s=s, c=PAPER, alpha=0.35 + 0.6 * w, linewidths=0)
    else:
        ax.scatter(gx, gy, s=s, c=[RAMP[min(9, int(x * 9.99))] for x in w], linewidths=0)
    ax.set_xlim(-1, 14); ax.set_ylim(-1, 14); ax.set_aspect("equal")


spot("intervals", intervals)
spot("dag", dag)
spot("contours", contours)
spot("field", field)
print("ok")
