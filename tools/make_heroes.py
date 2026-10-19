"""Decorative hero images in the Ink & Cobalt palette (abstract, not real data)."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
PAPER, INK, COBALT, DEEP, TINT, RULE = "#FCFCFB", "#121417", "#2F5BD3", "#1E43A8", "#E6ECFB", "#DFE2E8"
RAMP = ["#EEF2FC", "#D9E2F8", "#BFCEF3", "#9FB4EC", "#7C98E4", "#5A7EDC", "#2F5BD3", "#2549B0", "#1A3687", "#0F1E3D"]
W, H, DPI = 24, 8, 100  # 2400 x 800 px


def canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, facecolor=PAPER)
    plt.close(fig)


rng = np.random.default_rng(7)

# 1. Contours: topographic lines, a nod to spatial ecology
fig, ax = canvas()
x = np.linspace(0, 3, 900)
y = np.linspace(0, 1, 300)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for _ in range(9):
    cx, cy = rng.uniform(0, 3), rng.uniform(-0.2, 1.2)
    s = rng.uniform(0.18, 0.5)
    Z += rng.uniform(0.5, 1.2) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2))
levels = np.linspace(Z.min(), Z.max(), 26)
cmap = LinearSegmentedColormap.from_list("c", ["#BFCEF3", COBALT, DEEP])
ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=1.6)
ax.contour(X, Y, Z, levels=levels[-3:-2], colors=[INK], linewidths=2.4)
ax.set_xlim(0, 3); ax.set_ylim(0, 1)
save(fig, "hero-contours.png")

# 2. Intervals: a day of 15-minute bars, shaped like an arrival curve
fig, ax = canvas()
n = 96 * 2
t = np.linspace(0, 2, n)
base = (np.exp(-((t % 1 - 0.42) ** 2) / 0.012) * 0.9 + np.exp(-((t % 1 - 0.62) ** 2) / 0.02) * 0.7 + 0.08)
vals = base * rng.uniform(0.85, 1.12, n)
vals = vals / vals.max() * 0.82
colors = [RAMP[min(9, 2 + int(v / 0.82 * 7.9))] for v in vals]
ax.bar(np.arange(n), vals, width=0.72, color=colors, bottom=0.06)
ax.plot(np.arange(n), base / base.max() * 0.82 * 1.0 + 0.06, color=INK, lw=2.2)
ax.axhline(0.06, color=RULE, lw=2)
ax.set_xlim(-1, n); ax.set_ylim(0, 1)
save(fig, "hero-intervals.png")

# 3. Field: dot grid with a density wave
fig, ax = canvas()
gx, gy = np.meshgrid(np.arange(0, 120), np.arange(0, 40))
gx = gx.ravel(); gy = gy.ravel()
wave = 0.5 + 0.5 * np.sin(gx / 9.0 + np.cos(gy / 7.0) * 1.6) * np.cos(gy / 11.0)
size = 6 + 60 * wave ** 2
col = [RAMP[min(9, int(w * 9.99))] for w in wave]
ax.scatter(gx, gy, s=size, c=col, linewidths=0)
hot = np.argsort(-wave)[:6]
ax.scatter(gx[hot], gy[hot], s=size[hot] * 1.3, c=INK, linewidths=0)
ax.set_xlim(-1, 120); ax.set_ylim(-1, 40)
save(fig, "hero-field.png")

# Headshot placeholder (replace with images/headshot.jpg, square, 800 px+)
fig = plt.figure(figsize=(8, 8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
fig.patch.set_facecolor(TINT); ax.set_facecolor(TINT)
ax.add_patch(plt.Circle((0.5, 0.62), 0.17, color="#BFCEF3"))
ax.add_patch(matplotlib.patches.Ellipse((0.5, 0.12), 0.62, 0.5, color="#BFCEF3"))
ax.text(0.5, 0.93, "your headshot here", ha="center", va="center", fontsize=22, color=DEEP, family="DejaVu Sans")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(os.path.join(OUT, "headshot.jpg"), dpi=100, facecolor=TINT, pil_kwargs={"quality": 90})
plt.close(fig)
print("ok")
