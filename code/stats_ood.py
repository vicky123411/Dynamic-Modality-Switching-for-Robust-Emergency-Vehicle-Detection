"""Clustered-bootstrap CIs and McNemar tests for the unseen-source (OOD) experiment,
merged with the main tests so that the Holm correction covers all of them."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_ood as O
import ev_v3 as V
import evcore as E

MAIN = O.MAIN; OOD = O.OOD; ST = MAIN / 'stats'
SEEDS = (0, 1, 2); GATES = ('proposed', 'proposed_mlp')


def main():
    per = {'ood_composed': {}, 'ood_real': {}}
    pc = pd.read_csv(OOD / 'pairs_oodc_test.csv'); g = pd.read_csv(OOD / 'ood_vgg.csv')
    for s in SEEDS:
        sid = V.score_sources(s, [], [], 'id')
        d = {k: V.assemble(pd.read_csv(OOD / f'pairs_id_{k}.csv'), sid, 'image_path', 'audio_path', 'label', 'scenario')
             for k in ('train', 'val')}
        F = V.fit_fusers(d['train'], d['val'], s)
        c = V.assemble(pc, V.score_sources(s, [], [], 'oodc'), 'image_path', 'audio_path', 'label', 'scenario')
        r = V.assemble(g, V.score_sources(s, [], [], 'oodr'), 'frame', 'audio', 'label', 'class')
        per['ood_composed'][s] = (V.score_all(F, c), F['thr'], c.y)
        per['ood_real'][s] = (V.score_all(F, r), F['thr'], r.y)
    ci = json.load(open(ST / 'auprc_diff_ci95.json'))
    tests = pd.read_csv(ST / 'mcnemar_gate_vs_noisyor.csv')
    tests = tests[~tests.benchmark.isin(per)][['benchmark', 'seed', 'gate', 'gate_right_baseline_wrong',
                                                 'gate_wrong_baseline_right', 'p']]
    new = []
    for bench, P in per.items():
        y = P[0][2]; rng = np.random.default_rng(2026)
        if bench == 'ood_composed':
            ii = pd.factorize(pc.image_path)[0]; aa = pd.factorize(pc.audio_path)[0]
        else:
            vv = pd.factorize(g.ytid)[0]
        diffs = {k: [] for k in GATES}
        for _ in range(1000):
            if bench == 'ood_composed':
                w = rng.multinomial(ii.max() + 1, np.full(ii.max() + 1, 1 / (ii.max() + 1)))[ii] * \
                    rng.multinomial(aa.max() + 1, np.full(aa.max() + 1, 1 / (aa.max() + 1)))[aa]
            else:
                w = rng.multinomial(vv.max() + 1, np.full(vv.max() + 1, 1 / (vv.max() + 1)))[vv]
            k_ = w > 0
            for gt in GATES:
                diffs[gt].append(np.mean([average_precision_score(P[s][2][k_], P[s][0][gt][k_], sample_weight=w[k_]) -
                                          average_precision_score(P[s][2][k_], P[s][0]['noisy_or'][k_], sample_weight=w[k_])
                                          for s in SEEDS]))
        ap = {m: float(np.mean([average_precision_score(P[s][2], P[s][0][m]) for s in SEEDS])) for m in ('noisy_or',) + GATES}
        ci[bench] = {'auprc': ap}
        for gt in GATES:
            ci[bench][gt] = {'diff': ap[gt] - ap['noisy_or'], 'ci95': [float(x) for x in np.percentile(diffs[gt], [2.5, 97.5])]}
        for s in SEEDS:
            S, thr, yy = P[s]
            pb = (S['noisy_or'] >= thr['noisy_or']).astype(int)
            for gt in GATES:
                m = E.mcnemar(yy, pb, (S[gt] >= thr[gt]).astype(int))
                new.append({'benchmark': bench, 'seed': s, 'gate': gt, 'gate_right_baseline_wrong': m['n01'],
                            'gate_wrong_baseline_right': m['n10'], 'p': m['p_value']})
            pd.DataFrame({'y': yy, **S}).to_csv(ST / f'predictions_{bench}_s{s}.csv', index=False, float_format='%.6f')
    t = pd.concat([tests, pd.DataFrame(new)]).sort_values('p').reset_index(drop=True)
    m = len(t); adj, run = [], 0.0
    for i, p in enumerate(t.p):
        run = max(run, min(1.0, (m - i) * p)); adj.append(run)
    t['p_holm'] = adj; t['significant_holm'] = t.p_holm < 0.05
    t['direction'] = np.where(t.gate_right_baseline_wrong > t.gate_wrong_baseline_right, 'gate better', 'noisy-OR better')
    t.sort_values(['benchmark', 'gate', 'seed']).to_csv(ST / 'mcnemar_gate_vs_noisyor.csv', index=False)
    json.dump(ci, open(ST / 'auprc_diff_ci95.json', 'w'), indent=1)
    print(json.dumps({k: ci[k] for k in per}, indent=1))
    print(t[t.benchmark.isin(per)].to_string()); print('total tests', m)


if __name__ == '__main__':
    main()
