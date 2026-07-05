"""Plot evaluation results: robustness across weather + speed-accuracy trade-off."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os

# --- Paths ---
proj_root = os.path.join(os.path.dirname(__file__), '..')
csv_path = os.path.join(proj_root, 'output/full_eval/result.csv')
out_dir = os.path.join(proj_root, 'output/full_eval')
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
per_cond = df[(df['condition'] != 'OVERALL') & (df['valid_iters'] == 4)].copy()
overall = df[(df['condition'] == 'OVERALL') & (df['valid_iters'] == 4)].copy()

# --- Palette ---
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'

CAT_4 = ['#2a78d6', '#1baf7a', '#eda100', '#008300']  # blue, aqua, yellow, green
MODEL_ORDER = ['20-30-48', '23-36-37', '20-26-39', '15-44-51']
MODEL_COLOR = {m: CAT_4[i] for i, m in enumerate(MODEL_ORDER)}

# Runtime from README (ms on 3090, 640×480, iters=4)
RUNTIME = {
    '20-30-48': 29.3,
    '23-36-37': 41.1,
    '20-26-39': 37.5,
    '15-44-51': None,  # not in README table
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['system-ui', 'DejaVu Sans', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK_SECONDARY,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'grid.color': GRIDLINE,
    'grid.linewidth': 0.5,
    'text.color': INK_PRIMARY,
})


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linewidth=0.5, color=GRIDLINE)


# ============================================================
# FIGURE 1: Robustness across weather conditions (line chart)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
style_ax(ax)

# Order conditions by difficulty (mean D1-all across models)
cond_difficulty = per_cond.groupby('condition')['d1_all'].mean().sort_values()
COND_ORDER = list(cond_difficulty.index)

for i, m in enumerate(MODEL_ORDER):
    vals = []
    for c in COND_ORDER:
        row = per_cond[(per_cond['model'] == m) & (per_cond['condition'] == c)]
        vals.append(row['d1_all'].values[0] if len(row) else 0)

    marker = 'o' if m != '15-44-51' else 's'
    lw = 2.2 if m == '20-30-48' else 1.5
    alpha = 1.0 if m == '20-30-48' else 0.7
    z = 10 if m == '20-30-48' else 1

    ax.plot(COND_ORDER, vals, marker=marker, linewidth=lw, markersize=7,
            color=MODEL_COLOR[m], label=m, alpha=alpha, zorder=z,
            markerfacecolor='white' if m != '20-30-48' else MODEL_COLOR[m],
            markeredgewidth=1.8, markeredgecolor=MODEL_COLOR[m])

    # Direct-label the endpoints
    ax.annotate(f'{m}\n{vals[-1]:.1f}%', xy=(COND_ORDER[-1], vals[-1]),
                xytext=(12, 0), textcoords='offset points', fontsize=8,
                color=MODEL_COLOR[m], va='center', fontweight='bold',
                alpha=alpha)

ax.set_ylabel('D1-all (%)', color=INK_SECONDARY)
ax.set_title('Robustness Across Weather Conditions (iters=4)',
             color=INK_PRIMARY, fontweight='bold', pad=12)
ax.set_ylim(bottom=0)

# Shade the gap between best and worst to highlight robustness spread
best = [per_cond[per_cond['condition'] == c]['d1_all'].min() for c in COND_ORDER]
worst = [per_cond[per_cond['condition'] == c]['d1_all'].max() for c in COND_ORDER]
ax.fill_between(range(len(COND_ORDER)), best, worst, alpha=0.06,
                color=INK_MUTED, label='spread (best→worst)')

ax.legend(frameon=False, fontsize=9, ncol=2,
          labelcolor=INK_SECONDARY, loc='upper left')

plt.tight_layout()
out = os.path.join(out_dir, 'chart_robustness.png')
fig.savefig(out, dpi=150, facecolor=SURFACE, edgecolor='none')
plt.close(fig)
print(f'Saved: {out}')

# ============================================================
# FIGURE 2: Speed-Accuracy Trade-off (scatter)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
style_ax(ax)

for m in MODEL_ORDER:
    rt = RUNTIME.get(m)
    row = overall[overall['model'] == m]
    d1 = row['d1_all'].values[0] if len(row) else None

    if rt is None or d1 is None:
        continue

    ax.scatter(rt, d1, s=180, c=MODEL_COLOR[m], edgecolors='white',
               linewidth=2, zorder=5, alpha=0.9)

    # Label: position text to avoid overlap
    offset_y = 0.12 if m != '23-36-37' else -0.22
    offset_x = 0 if m != '20-26-39' else -1.8
    ax.annotate(m, (rt, d1), textcoords='offset points',
                xytext=(offset_x * 15, offset_y * 50), fontsize=9,
                color=MODEL_COLOR[m], fontweight='bold', ha='center')

# Best-in-class crossover line (Pareto frontier)
labeled = [(RUNTIME[m], overall[overall['model'] == m]['d1_all'].values[0])
           for m in MODEL_ORDER if RUNTIME.get(m) is not None]
labeled.sort()  # by speed
pareto_rt, pareto_d1 = [], []
best_d1 = float('inf')
for rt, d1 in labeled:
    if d1 < best_d1:
        pareto_rt.append(rt)
        pareto_d1.append(d1)
        best_d1 = d1
ax.plot(pareto_rt, pareto_d1, '--', color=INK_MUTED, linewidth=1.2, alpha=0.6)

ax.set_xlabel('Runtime on 3090 (ms, lower is faster)', color=INK_SECONDARY)
ax.set_ylabel('D1-all (%)', color=INK_SECONDARY)
ax.set_title('Speed–Accuracy Trade-off (iters=4)',
             color=INK_PRIMARY, fontweight='bold', pad=12)

# Invert x-axis so "faster = right" feels natural; or keep "less is better" convention
ax.invert_xaxis()  # faster → right

plt.tight_layout()
out = os.path.join(out_dir, 'chart_speed_vs_accuracy.png')
fig.savefig(out, dpi=150, facecolor=SURFACE, edgecolor='none')
plt.close(fig)
print(f'Saved: {out}')

print('\nDone.')
