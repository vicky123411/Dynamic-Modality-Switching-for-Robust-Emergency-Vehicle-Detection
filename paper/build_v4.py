"""Build main_v4.tex from the result JSON files (every number in the paper comes from here)."""
import json, re, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.environ.get('EV_RESULTS', os.path.join(HERE, '..', 'results'))
SEEDS = [s for s in (0, 1, 2) if os.path.exists(f'{W}/results_s{s}.json')]
R = [json.load(open(f'{W}/results_s{s}.json')) for s in SEEDS]
N = {}


def ms(proto, m, k):
    v = np.array([r[proto]['test'][m][k] for r in R], float)
    return v.mean(), v.std()


def f3(x):
    return f'{x:.3f}'


def pm(proto, m, k, bold=False, it=False):
    a, b = ms(proto, m, k)
    s = f'{a:.3f}'
    if bold: s = r'\textbf{' + s + '}'
    return s + r'$\pm$' + f'{b:.3f}'


METHODS = [('Vision only', 'vision_only'), ('Audio only', 'audio_only'), ('Mean', 'mean'), ('Max', 'max'),
           ('Noisy-OR', 'noisy_or'), ('Static weighted', 'static'), ('Learned late fusion', 'learned_late'),
           ('Hand-crafted rules', 'rule_context'), ('Gate (logistic)', 'proposed'), ('Gate (MLP)', 'proposed_mlp'),
           ('Oracle gate (UB)', 'oracle')]
NONORACLE = [k for _, k in METHODS if k != 'oracle']


def best_of(proto, k):
    return max(NONORACLE, key=lambda m: ms(proto, m, k)[0])


def isbest(proto, m, c):
    return m != 'oracle' and round(ms(proto, m, c)[0], 3) == round(ms(proto, best_of(proto, c), c)[0], 3)


def table_rows(proto, cols):
    out = []
    for name, m in METHODS:
        if m in ('mean', 'proposed', 'oracle'):
            out.append(r'\midrule')
        it = m == 'oracle'
        cells = [pm(proto, m, c, bold=isbest(proto, m, c)) for c in cols]
        nm = r'\textit{' + name + '}' if it else name
        out.append(nm + ' & ' + ' & '.join(cells) + r' \\')
    return '\n'.join(out)


def table_rows_real(cols):
    out = []
    for name, m in METHODS:
        if m in ('mean', 'proposed', 'oracle'):
            out.append(r'\midrule')
        it = m == 'oracle'
        cells = [pm('vgg_indomain', m, c, bold=isbest('vgg_indomain', m, c)) for c in cols] + \
                [pm('vgg_zeroshot', m, c, bold=isbest('vgg_zeroshot', m, c)) for c in cols]
        nm = r'\textit{' + name + '}' if it else name
        out.append(nm + ' & ' + ' & '.join(cells) + r' \\')
    return '\n'.join(out)


# ------------------------------------------------------------------ numbers for the prose
S = 'synthetic_pair42'
for proto, tag in ((S, 'syn'), ('vgg_indomain', 'ind'), ('vgg_zeroshot', 'zs')):
    for m in [k for _, k in METHODS]:
        for k in ('auprc', 'auroc', 'f1', 'recall_at_fpr10', 'precision', 'recall'):
            a, b = ms(proto, m, k)
            N[f'{tag}.{m}.{k}'] = f3(a)
            N[f'{tag}.{m}.{k}.sd'] = f3(b)
fus = ['mean', 'max', 'noisy_or', 'static', 'learned_late', 'proposed', 'proposed_mlp']
for tag, proto in (('syn', S), ('ind', 'vgg_indomain'), ('zs', 'vgg_zeroshot')):
    v = [ms(proto, m, 'auprc')[0] for m in fus]
    N[f'{tag}.fus.min'] = f3(min(v)); N[f'{tag}.fus.max'] = f3(max(v))
    v = [ms(proto, m, 'recall_at_fpr10')[0] for m in fus]
    N[f'{tag}.r10.min'] = f3(min(v)); N[f'{tag}.r10.max'] = f3(max(v))

# pairing seeds (training seed 0..2 averaged): AUPRC of noisy-or / gates per pairing seed
for ps in (42, 43, 44):
    for m in ('noisy_or', 'proposed', 'proposed_mlp', 'max'):
        N[f'pair{ps}.{m}'] = f3(np.mean([r[f'synthetic_pair{ps}']['test'][m]['auprc'] for r in R]))
# McNemar counts
def sig_count(proto, g):
    c = sum(r[proto]['sig'][g]['significant'] for r in R)
    fav = sum(r[proto]['sig'][g]['significant'] and r[proto]['sig'][g]['n01'] > r[proto]['sig'][g]['n10'] for r in R)
    ps = [r[proto]['sig'][g]['p_value'] for r in R]
    return c, fav, ps
sigs = {}
for proto in [f'synthetic_pair{p}' for p in (42, 43, 44)] + ['vgg_indomain', 'vgg_zeroshot']:
    for g in ('proposed', 'proposed_mlp'):
        sigs[(proto, g)] = sig_count(proto, g)
N['sig.syn.total'] = str(len(R) * 3)
N['sig.syn.log'] = str(sum(sigs[(f'synthetic_pair{p}', 'proposed')][1] for p in (42, 43, 44)))
N['sig.syn.mlp'] = str(sum(sigs[(f'synthetic_pair{p}', 'proposed_mlp')][1] for p in (42, 43, 44)))
N['sig.syn.log.any'] = str(sum(sigs[(f'synthetic_pair{p}', 'proposed')][0] for p in (42, 43, 44)))
N['sig.syn.mlp.any'] = str(sum(sigs[(f'synthetic_pair{p}', 'proposed_mlp')][0] for p in (42, 43, 44)))
N['sig.ind.log'] = str(sigs[('vgg_indomain', 'proposed')][1]); N['sig.ind.mlp'] = str(sigs[('vgg_indomain', 'proposed_mlp')][1])
N['sig.zs.log'] = str(sigs[('vgg_zeroshot', 'proposed')][1]); N['sig.zs.mlp'] = str(sigs[('vgg_zeroshot', 'proposed_mlp')][1])
N['nseeds'] = str(len(R))
N['sig.syn.log.worse'] = str(sum(sigs[(f'synthetic_pair{p}', 'proposed')][0] - sigs[(f'synthetic_pair{p}', 'proposed')][1] for p in (42, 43, 44)))
N['sig.syn.mlp.worse'] = str(sum(sigs[(f'synthetic_pair{p}', 'proposed_mlp')][0] - sigs[(f'synthetic_pair{p}', 'proposed_mlp')][1] for p in (42, 43, 44)))
N['sig.zs.log.any'] = str(sigs[('vgg_zeroshot', 'proposed')][0]); N['sig.zs.mlp.any'] = str(sigs[('vgg_zeroshot', 'proposed_mlp')][0])
N['sig.ind.log.any'] = str(sigs[('vgg_indomain', 'proposed')][0]); N['sig.ind.mlp.any'] = str(sigs[('vgg_indomain', 'proposed_mlp')][0])
def gap(proto, g, b='noisy_or'):
    return np.mean([r[proto]['test'][g]['auprc'] - r[proto]['test'][b]['auprc'] for r in R])
gs = [gap(f'synthetic_pair{p}', g) for p in (42, 43, 44) for g in ('proposed', 'proposed_mlp')]
N['gap.syn.max'] = f'{max(gs):.3f}'; N['gap.syn.min'] = f'{min(gs):.3f}'
N['nruns'] = str(3 * len(R))
for tag, proto in (('ind', 'vgg_indomain'), ('zs', 'vgg_zeroshot')):
    for g, gg in (('log', 'proposed'), ('mlp', 'proposed_mlp')):
        c, f, _ = sigs[(proto, gg)]
        N[f'sig.{tag}.{g}.worse'] = str(c - f)

# calibration
for tag, proto in (('syn', S), ('ind', 'vgg_indomain')):
    for mod, i in (('v', 0), ('a', 1)):
        N[f'T.{tag}.{mod}'] = f'{np.mean([r[proto]["T"][i] for r in R]):.2f}'
        nm = 'vision' if mod == 'v' else 'audio'
        N[f'ece0.{tag}.{mod}'] = f3(np.mean([r[proto]['ece']['before'][nm] for r in R]))
        N[f'ece1.{tag}.{mod}'] = f3(np.mean([r[proto]['ece']['after'][nm] for r in R]))
    N[f'wv.{tag}'] = f'{np.mean([r[proto]["static_wv"] for r in R]):.2f}'
    N[f'gwv.{tag}'] = f'{np.mean([r[proto]["gate"]["mean_wv"] for r in R]):.2f}'
    N[f'gwvsd.{tag}'] = f'{np.mean([r[proto]["gate"]["std_wv"] for r in R]):.2f}'

# gate coefficients (mean over seeds)
def coefs(proto):
    keys = R[0][proto]['gate']['coef'].keys()
    return {k: float(np.mean([r[proto]['gate']['coef'][k] for r in R])) for k in keys}
C_syn, C_ind = coefs(S), coefs('vgg_indomain')
NICE = {'p_vision': r'$\hat p_v$', 'p_audio': r'$\hat p_a$', 'conf_margin_v': 'margin (vision)',
        'conf_margin_a': 'margin (audio)', 'luminance': 'mean luminance', 'rms_contrast': 'RMS contrast',
        'lap_var': 'Laplacian variance', 'dark_channel': 'dark channel', 'saturated_frac': 'saturated pixels',
        'edge_density': 'edge density', 'colorfulness': 'colourfulness', 'rms_db': 'RMS level',
        'est_snr_db': 'SNR estimate', 'spectral_flatness': 'spectral flatness', 'siren_band_ratio': 'siren-band energy',
        'harmonic_ratio': 'top-1\\% energy share', 'spectral_centroid': 'spectral centroid'}
top = sorted(set(sorted(C_syn, key=lambda k: -abs(C_syn[k]))[:5]) | set(sorted(C_ind, key=lambda k: -abs(C_ind[k]))[:5]),
             key=lambda k: -max(abs(C_syn[k]), abs(C_ind[k])))
N['coef.rows'] = '\n'.join(f'{NICE[k]} & ${C_syn[k]:+.2f}$ & ${C_ind[k]:+.2f}$ \\\\' for k in top)
for k in C_syn:
    N[f'cs.{k}'] = f'{C_syn[k]:+.2f}'; N[f'ci.{k}'] = f'{C_ind[k]:+.2f}'
qual = [k for k in C_syn if k not in ('p_vision', 'p_audio', 'conf_margin_v', 'conf_margin_a')]
N['cs.qualmax'] = f'{max(abs(C_syn[k]) for k in qual):.2f}'

# dark / bright real frames
for nm in ('dark', 'bright'):
    vals = {}
    for m in ('vision_only', 'audio_only', 'noisy_or', 'proposed', 'proposed_mlp', 'static'):
        vals[m] = np.mean([r['vgg_dark'][nm]['res'][m]['auprc'] for r in R])
        N[f'{nm}.{m}'] = f3(vals[m])
    N[f'{nm}.n'] = str(R[0]['vgg_dark'][nm]['n']); N[f'{nm}.pos'] = str(R[0]['vgg_dark'][nm]['pos'])
# image-only test
N['vt.auprc'] = f3(np.mean([r['vision_test']['auprc'] for r in R]))
N['vt.in'] = f3(np.mean([r['vision_test']['by_source']['imagenet']['auprc'] for r in R]))
N['vt.oi'] = f3(np.mean([r['vision_test']['by_source']['openimages']['auprc'] for r in R]))
N['vt.adv'] = f3(np.mean([r['vision_test']['adverse_tagged_or_dark']['adverse']['auprc'] for r in R]))
N['vt.oth'] = f3(np.mean([r['vision_test']['adverse_tagged_or_dark']['other']['auprc'] for r in R]))
N['vt.adv.n'] = str(R[0]['vision_test']['adverse_tagged_or_dark']['adverse']['n'])
N['vt.adv.pos'] = str(R[0]['vision_test']['adverse_tagged_or_dark']['adverse']['pos'])

# sweep
SW = [d for d in (json.load(open(f"{W}/sweep_s{s}.json")) for s in SEEDS if os.path.exists(f"{W}/sweep_s{s}.json")) if len(d) == 41]
def sw(key, m):
    return np.mean([d[key][m] for d in SW])
N['sw.n'] = str(len(SW))
N['sw.clean.v'] = f'{sw("clean|None|0", "vision_only"):.2f}'
N['sw.clean.a'] = f'{sw("clean|None|0", "audio_only"):.2f}'
N['sw.fog5.v'] = f'{sw("visual|fog|5", "vision_only"):.2f}'
N['sw.glare5.v'] = f'{sw("visual|glare|5", "vision_only"):.2f}'
N['sw.ll5.v'] = f'{sw("visual|low_light|5", "vision_only"):.2f}'
fm = ['static', 'noisy_or', 'proposed']
N['sw.vis5.fusmin'] = f'{min(sw(f"visual|{c}|5", m) for c in ("low_light","fog","rain","motion_blur","glare") for m in fm):.2f}'
N['sw.tn5.a'] = f'{sw("acoustic|traffic_noise|5", "audio_only"):.2f}'
N['sw.aud5.fusmin'] = f'{min(sw(f"acoustic|{c}|5", m) for c in ("traffic_noise","wind_noise","clipping") for m in fm):.2f}'
N['sw.fog5.wv'] = f'{sw("visual|fog|5", "mean_wv"):.2f}'
N['sw.ll5.wv'] = f'{sw("visual|low_light|5", "mean_wv"):.2f}'
N['sw.tn5.wv'] = f'{sw("acoustic|traffic_noise|5", "mean_wv"):.2f}'
N['sw.clean.wv'] = f'{sw("clean|None|0", "mean_wv"):.2f}'

# ---- clustered-bootstrap CIs and Holm-corrected McNemar (stats_v4.py)
import pandas as pd
CI = json.load(open(f'{W}/stats/auprc_diff_ci95.json'))
MC = pd.read_csv(f'{W}/stats/mcnemar_gate_vs_noisyor.csv')
def wl(bench, g):
    t = MC[(MC.benchmark == bench) & (MC.gate == g) & MC.significant_holm]
    return int((t.direction == 'gate better').sum()), int((t.direction != 'gate better').sum())
def cis(bench, g):
    d = CI[bench][g]; lo, hi = d['ci95']
    f = lambda x: ('$+$' if x >= 0 else '$-$') + f'{abs(x):.4f}'
    return f"{f(d['diff'])} [{f(lo)}, {f(hi)}]"
rows5 = []
for bench, name in (('composed_pair42', 'EV-AVSim-L, pairing 42'), ('composed_pair43', 'EV-AVSim-L, pairing 43'),
                    ('composed_pair44', 'EV-AVSim-L, pairing 44'), ('real_indomain', 'VGGSound-EV, in-domain'),
                    ('real_zeroshot', 'VGGSound-EV, zero-shot'), ('ood_composed', 'Out-of-source, composite'),
                    ('ood_real', 'Out-of-source, real clips')):
    cells = []
    for g in ('proposed', 'proposed_mlp'):
        w_, l_ = wl(bench, g); cells += [cis(bench, g), f'{w_}/{l_}']
    rows5.append(name + ' & ' + ' & '.join(cells) + r' \\')
N['tab.stats'] = '\n'.join(rows5)
for g, tag in (('proposed', 'log'), ('proposed_mlp', 'mlp')):
    N[f'h.syn.{tag}.win'] = str(sum(wl(f'composed_pair{p}', g)[0] for p in (42, 43, 44)))
    N[f'h.syn.{tag}.loss'] = str(sum(wl(f'composed_pair{p}', g)[1] for p in (42, 43, 44)))
    for b, bt in (('real_indomain', 'ind'), ('real_zeroshot', 'zs')):
        N[f'h.{bt}.{tag}.win'], N[f'h.{bt}.{tag}.loss'] = map(str, wl(b, g))
N['ntests'] = str(len(MC))
# ---- out-of-distribution experiment (ev_ood.py)
WO = os.environ.get('EV_OOD', os.path.join(HERE, '..', 'results_ood'))
RO = [json.load(open(f'{WO}/results_ood_s{s}.json')) for s in (0, 1, 2) if os.path.exists(f'{WO}/results_ood_s{s}.json')]
if RO:
    def om(split, m, k):
        v = np.array([r[split][m][k] for r in RO], float); return v.mean(), v.std()
    OM = [('Vision only', 'vision_only'), ('Audio only', 'audio_only'), ('Mean', 'mean'), ('Max', 'max'),
          ('Noisy-OR', 'noisy_or'), ('Static weighted', 'static'), ('Learned late fusion', 'learned_late'),
          ('Gate (logistic)', 'proposed'), ('Gate (MLP)', 'proposed_mlp')]
    cols = [('id', 'auprc', True), ('ood_composed', 'auprc', True), ('ood_real', 'auprc', True),
            ('ood_real', 'f1', False), ('ood_real', 'fpr', False), ('ood_real', 'ece', False), ('ood_real', 'brier', False)]
    best = {}
    for sp, k, _ in cols:
        vals = {m: round(om(sp, m, k)[0], 3) for _, m in OM}
        best[(sp, k)] = (min if k in ('fpr', 'ece', 'brier') else max)(vals.values())
    rows = []
    for name, m in OM:
        if m == 'mean': rows.append(r'\midrule')
        if m == 'proposed': rows.append(r'\midrule')
        cells = []
        for sp, k, sd in cols:
            a_, b_ = om(sp, m, k)
            c = f'{a_:.3f}'
            if round(a_, 3) == best[(sp, k)]: c = r'\textbf{' + c + '}'
            if sd: c += r'$\pm$' + f'{b_:.3f}'
            cells.append(c)
        rows.append(name + ' & ' + ' & '.join(cells) + r' \\')
    N['tab.ood'] = '\n'.join(rows)
    for sp, tag in (('id', 'oid'), ('ood_composed', 'oc'), ('ood_real', 'or')):
        for _, m in OM:
            for k in ('auprc', 'f1', 'fpr', 'ece', 'brier', 'recall'):
                N[f'{tag}.{m}.{k}'] = f3(om(sp, m, k)[0])
        N[f'{tag}.n'] = f"{RO[0][sp]['n']:,}"; N[f'{tag}.pos'] = f"{RO[0][sp]['pos']:,}"
        fz = [om(sp, m, 'auprc')[0] for m in ('mean', 'max', 'noisy_or', 'static', 'learned_late', 'proposed', 'proposed_mlp')]
        N[f'{tag}.fus.min'] = f3(min(fz)); N[f'{tag}.fus.max'] = f3(max(fz))
    for k in ('vision_id_auprc', 'vision_ood_auprc', 'audio_id_auprc', 'audio_ood_auprc'):
        N[f'ob.{k}'] = f3(np.mean([r['branch'][k] for r in RO]))
    N['ob.n_ood_images'] = f"{RO[0]['branch']['n_ood_images']:,}"
    N['nseeds_ood'] = str(len(RO))
    D = json.load(open(f'{WO}/duplicates.json'))
    N['dup.img'] = str(D['oi_images_near_dup']); N['dup.frame'] = str(D['vgg_frames_near_dup']); N['dup.aud'] = str(D['vgg_audio_near_dup'])
N['tab.syn'] = table_rows(S, ['auprc', 'auroc', 'f1', 'recall_at_fpr10'])
N['tab.real'] = table_rows_real(['auprc', 'f1', 'recall_at_fpr10'])

D = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(D, 'main_v4_template.tex')).read()
N['FIG1'] = open(os.path.join(D, 'fig1_block.tex')).read()
bib = json.load(open(os.path.join(D, 'bib_v4.json')))
order = []
for keys in re.findall(r'\\cite\{([^}]+)\}', tpl):
    for k in keys.split(','):
        k = k.strip()
        if k not in order:
            order.append(k)
unused = set(bib) - set(order); undefined = set(order) - set(bib)
if unused or undefined:
    sys.exit(f'bib mismatch unused={unused} undefined={undefined}')
N['BIB'] = '\\begin{thebibliography}{99}\n\n' + '\n\n'.join('\\bibitem{%s}\n%s' % (k, bib[k]) for k in order) + '\n\n\\end{thebibliography}'
missing = set(re.findall(r'@@([^@]+)@@', tpl)) - set(N)
if missing:
    sys.exit('missing keys: %s' % sorted(missing))
out = re.sub(r'@@([^@]+)@@', lambda m: N[m.group(1)], tpl)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_v4.tex'), 'w').write(out)
json.dump(N, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numbers_v4.json'), 'w'), indent=1)
print('seeds', SEEDS, 'sweeps', len(SW), 'ok')
