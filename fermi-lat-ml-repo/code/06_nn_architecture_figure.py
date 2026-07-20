#!/usr/bin/env python3
"""
06_nn_architecture_figure.py — Draw the exact MLP architecture used in
03_train_classify.py: 10 -> 128 -> 64 -> 32 -> 4, ReLU activations,
Adam optimiser, early stopping. Purely illustrative (no data dependency).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

layers = [10, 128, 64, 32, 4]
layer_names = ['Input\n(10 features)', 'Hidden 1\n(128, ReLU)',
               'Hidden 2\n(64, ReLU)', 'Hidden 3\n(32, ReLU)',
               'Output\n(4, softmax)']
# cap the number of drawn nodes per layer for readability; show "..." for large layers
max_draw = 14

fig, ax = plt.subplots(figsize=(11, 6.2))

colors = ['#4C72B0', '#DD8452', '#DD8452', '#DD8452', '#55A868']
x_positions = np.linspace(0.06, 0.94, len(layers))

node_positions = []  # list of lists of (x,y) per layer
for li, n in enumerate(layers):
    n_draw = min(n, max_draw)
    ys = np.linspace(0.12, 0.88, n_draw)
    node_positions.append(list(zip([x_positions[li]]*n_draw, ys)))

# draw connections (subsampled for the large layers, to keep it legible)
rng = np.random.default_rng(0)
for li in range(len(layers)-1):
    src = node_positions[li]
    dst = node_positions[li+1]
    for (x0, y0) in src:
        # connect to a subset of destination nodes if there are many, else all
        if len(dst) > 8:
            idx = rng.choice(len(dst), size=6, replace=False)
        else:
            idx = range(len(dst))
        for j in idx:
            x1, y1 = dst[j]
            ax.plot([x0, x1], [y0, y1], color='grey', alpha=0.15, lw=0.6, zorder=1)

# draw nodes
for li, (n, name) in enumerate(zip(layers, layer_names)):
    pts = node_positions[li]
    for (x, y) in pts:
        circ = plt.Circle((x, y), 0.016, color=colors[li], ec='white',
                           lw=0.8, zorder=3)
        ax.add_patch(circ)
    if n > max_draw:
        ax.text(x_positions[li], 0.02, f'$\\vdots$\n$N={n}$', ha='center',
                 va='top', fontsize=10)
    else:
        ax.text(x_positions[li], 0.02, f'$N={n}$', ha='center', va='top',
                 fontsize=10)
    ax.text(x_positions[li], 0.96, name, ha='center', va='bottom',
             fontsize=11, fontweight='bold')

# annotate activation / optimiser details between layers
annots = ['StandardScaler\n(z-score)', 'ReLU', 'ReLU', 'ReLU', 'Softmax']
for li in range(1, len(layers)):
    xmid = (x_positions[li-1] + x_positions[li]) / 2
    ax.text(xmid, 0.94, annots[li], ha='center', va='bottom', fontsize=8.5,
             style='italic', color='#444444')

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.08, 1.02)
ax.axis('off')
ax.set_title('Multi-Layer Perceptron architecture ($10\\to128\\to64\\to32\\to4$)\n'
              'Adam optimiser, early stopping, $\\alpha=10^{-4}$; 12{,}708 trainable parameters',
              fontsize=12)

legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0',
           markersize=10, label='Input layer (standardised features)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#DD8452',
           markersize=10, label='Hidden layers (ReLU activation)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#55A868',
           markersize=10, label='Output layer (softmax, 4 classes)'),
]
ax.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False,
          bbox_to_anchor=(0.5, -0.06), fontsize=9)

plt.tight_layout()
plt.savefig('fig10_mlp_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig('fig10_mlp_architecture.pdf', bbox_inches='tight')
print("Saved fig10_mlp_architecture.png and .pdf")
