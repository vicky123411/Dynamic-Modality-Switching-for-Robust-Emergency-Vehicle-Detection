"""Extra statistics requested in review:
 - 95% confidence intervals for AUPRC(gate) - AUPRC(noisy-OR) with a CLUSTERED bootstrap
   (composed pairs: two-way resampling of source images and source audio clips;
    real clips: resampling of YouTube videos), averaged over the 3 training seeds;
 - exact McNemar p-values (gate vs noisy-OR, at validation thresholds) for every
   seed x benchmark, with Holm correction over all tests;
 - per-sample test predictions exported to CSV.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_v3 as V
import evcore as E

W = V.WORK
OUT = W / 'stats'; OUT.mkdir(exist_ok=True)
SEEDS = (0, 1, 2)
GATES = ('proposed', 'proposed_mlp')


def collect():
    rows = {}   # bench -> dict(seed -> (S, thr, y)), plus cluster ids
    meta = {}
    vgg = pd.read_csv(W / 'vgg_index.csv')
    for s in SEEDS:
        src = V.score_sources(s, [], [], 'synthetic')
        F42 = None
        for ps in (42, 43, 44):
            dfs = {k: pd.read_csv(W / f'pairs_seed{ps}_{k}.csv') for k in ('train', 'val', 'test')}
            d = {k: V.assemble(dfs[k], src, 'image_path', 'audio_path', 'label', 'scenario') for k in dfs}
            F = V.fit_fusers(d['train'], d['val'], s)
            S = V.score_all(F, d['test'])
            rows.setdefault(f'composed_pair{ps}', {})[s] = (S, F['thr'], d['test'].y)
            meta[f'composed_pair{ps}'] = dfs['test']
            if ps == 42:
                F42 = F
        vs = V.score_sources(s, [], [], 'vgg')
        dv = {k: V.assemble(vgg[vgg.split == k], vs, 'frame', 'audio', 'label', 'class') for k in ('train', 'val', 'test')}
        Fv = V.fit_fusers(dv['train'], dv['val'], s)
        rows.setdefault('real_indomain', {})[s] = (V.score_all(Fv, dv['test']), Fv['thr'], dv['test'].y)
        rows.setdefault('real_zeroshot', {})[s] = (V.score_all(F42, dv['test']), F42['thr'], dv['test'].y)
        meta['real_indomain'] = meta['real_zeroshot'] = vgg[vgg.split == 'test'].reset_index(drop=True)
    return rows, meta


def cluster_weights(meta, bench, rng):
    m = meta[bench]
    if bench.startswith('composed'):
        ii = pd.factorize(m.image_path)[0]; aa = pd.factorize(m.audio_path)[0]
        ci = rng.multinomial(ii.max() + 1, np.ones(ii.max() + 1) / (ii.max() + 1))
        ca = rng.multinomial(aa.max() + 1, np.ones(aa.max() + 1) / (aa.max() + 1))
        return ci[ii] * ca[aa]
    vv = pd.factorize(m.ytid)[0]
    cv = rng.multinomial(vv.max() + 1, np.ones(vv.max() + 1) / (vv.max() + 1))
    return cv[vv]


def main():
    rows, meta = collect()
    ci_out, tests = {}, []
    for bench, per in rows.items():
        y = per[0][2]
        # point estimates (mean over seeds)
        ap = {m: np.mean([average_precision_score(per[s][2], per[s][0][m]) for s in SEEDS])
              for m in ('noisy_or',) + GATES}
        rng = np.random.default_rng(2026)
        diffs = {g: [] for g in GATES}
        for _ in range(1000):
            w = cluster_weights(meta, bench, rng)
            keep = w > 0
            if len(set(y[keep])) < 2:
                continue
            for g in GATES:
                dd = [average_precision_score(per[s][2][keep], per[s][0][g][keep], sample_weight=w[keep]) -
                      average_precision_score(per[s][2][keep], per[s][0]['noisy_or'][keep], sample_weight=w[keep])
                      for s in SEEDS]
                diffs[g].append(np.mean(dd))
        ci_out[bench] = {'auprc': ap}
        for g in GATES:
            lo, hi = np.percentile(diffs[g], [2.5, 97.5])
            ci_out[bench][g] = {'diff': ap[g] - ap['noisy_or'], 'ci95': [float(lo), float(hi)]}
        for s in SEEDS:
            S, thr, yy = per[s]
            pb = (S['noisy_or'] >= thr['noisy_or']).astype(int)
            for g in GATES:
                pg = (S[g] >= thr[g]).astype(int)
                r = E.mcnemar(yy, pb, pg)
                tests.append({'benchmark': bench, 'seed': s, 'gate': g, 'gate_right_baseline_wrong': r['n01'],
                              'gate_wrong_baseline_right': r['n10'], 'p': r['p_value']})
            # export predictions
            df = pd.DataFrame({'y': yy, **{k: v for k, v in S.items()}})
            df.to_csv(OUT / f'predictions_{bench}_s{s}.csv', index=False, float_format='%.6f')
    # Holm correction
    t = pd.DataFrame(tests).sort_values('p').reset_index(drop=True)
    m = len(t); adj, run = [], 0
    for i, p in enumerate(t.p):
        run = max(run, min(1.0, (m - i) * p)); adj.append(run)
    t['p_holm'] = adj
    t['significant_holm'] = t.p_holm < 0.05
    t['direction'] = np.where(t.gate_right_baseline_wrong > t.gate_wrong_baseline_right, 'gate better', 'noisy-OR better')
    t.sort_values(['benchmark', 'gate', 'seed']).to_csv(OUT / 'mcnemar_gate_vs_noisyor.csv', index=False)
    json.dump(ci_out, open(OUT / 'auprc_diff_ci95.json', 'w'), indent=1)
    print(json.dumps(ci_out, indent=1))
    print(t.sort_values(['benchmark', 'gate', 'seed']).to_string())


if __name__ == '__main__':
    main()
