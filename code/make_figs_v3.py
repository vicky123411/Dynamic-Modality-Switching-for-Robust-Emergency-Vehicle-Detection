"""Degradation-curve figure from sweep JSON(s). usage: python make_figs_v3.py OUT.png sweep_s0.json [sweep_s1.json ...]"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out, files = sys.argv[1], sys.argv[2:]
runs = [json.load(open(f)) for f in files]
VIS = ['low_light', 'fog', 'rain', 'motion_blur', 'glare']
AUD = ['traffic_noise', 'wind_noise', 'clipping']
TITLE = {'low_light': 'low light', 'fog': 'fog', 'rain': 'rain', 'motion_blur': 'motion blur', 'glare': 'glare',
         'traffic_noise': 'traffic noise', 'wind_noise': 'wind noise', 'clipping': 'clipping'}
M = [('vision_only', 'Vision only', '#1f77b4', 'o'), ('audio_only', 'Audio only', '#2ca02c', 's'),
     ('static', 'Static weighted', '#9467bd', '^'), ('noisy_or', 'Noisy-OR', '#ff7f0e', 'D'),
     ('proposed', 'Gate (logistic)', '#d62728', 'v')]
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
panels = [('visual', c) for c in VIS] + [('acoustic', c) for c in AUD]
fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.3), sharey=True, sharex=True)
for ax, (ch, c) in zip(axes.flat, panels):
    for key, name, col, mk in M:
        ys = np.array([[r['clean|None|0'][key]] + [r[f'{ch}|{c}|{s}'][key] for s in range(1, 6)] for r in runs])
        mu, sd = ys.mean(0), ys.std(0)
        ax.plot(range(6), mu, color=col, marker=mk, ms=2.5, lw=1.1, label=name)
        if len(runs) > 1:
            ax.fill_between(range(6), mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    ax.set_title(TITLE[c] + (' (audio)' if ch == 'acoustic' else ''), fontsize=8, pad=2)
    ax.set_xticks(range(6)); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.tick_params(labelsize=7, pad=1)
for ax in axes[:, 0]:
    ax.set_ylabel('$F_1$', fontsize=8)
for ax in axes[1, :]:
    ax.set_xlabel('severity', fontsize=8, labelpad=1)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.4)
fig.savefig(out, dpi=600)
print('saved', out)
