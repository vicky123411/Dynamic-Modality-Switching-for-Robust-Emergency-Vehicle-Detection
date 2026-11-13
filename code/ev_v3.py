"""
EV detection v3: larger data + real synchronized audio-visual test.

Stages (each cached under WORK):
  index   build vision / audio / VGGSound-EV indices, leak-free splits, synthetic pairs
  vision  fine-tune MobileNetV3-Small on the enlarged image corpus
  audio   train the log-Mel CNN on UrbanSound8K folds 1-8 + VGGSound-EV train audio
  score   per-source confidences + 17-d descriptors (clean)
  eval    synthetic protocol (3 pairing seeds), real VGGSound-EV protocol (in-domain + zero-shot)
  sweep   synthetic corruption sweep (5 visual + 3 acoustic x 5 severities)
usage: python ev_v3.py STAGE [--seed S]
"""
import argparse, json, logging, os, sys, time, glob
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evcore as E

torch.set_num_threads(2)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("v3")
DATA = Path(os.environ.get('EV_DATA', 'data'))
WORK = Path(os.environ.get('EV_WORK', 'work'))
WORK.mkdir(parents=True, exist_ok=True)
SR = E.SR
POS_VGG = {'ambulance siren', 'fire truck siren', 'police car (siren)'}


# ----------------------------------------------------------------- indices
def split_strat(df, keycols, ratios=(0.7, 0.15, 0.15), seed=42, group=None):
    rng = np.random.default_rng(seed)
    df = df.copy(); df['split'] = ''
    for _, g in df.groupby(keycols):
        units = g[group].unique() if group else g.index.to_numpy()
        units = np.array(sorted(units)); rng.shuffle(units)
        n = len(units); a = int(round(ratios[0] * n)); b = a + int(round(ratios[1] * n))
        lab = {u: s for us, s in ((units[:a], 'train'), (units[a:b], 'val'), (units[b:], 'test')) for u in us}
        if group:
            df.loc[g.index, 'split'] = g[group].map(lab)
        else:
            df.loc[g.index, 'split'] = [lab[i] for i in g.index]
    return df


def load_us8k():
    us = E.load_urbansound_index(DATA / 'us8k/UrbanSound8K')
    cross = us.groupby('source_id').split.nunique()
    bad = set(cross[cross > 1].index)
    log.info('dropping %d slices from %d recording(s) that span two folds-splits: %s',
             int(us.source_id.isin(bad).sum()), len(bad), sorted(bad))
    us = us[~us.source_id.isin(bad)].reset_index(drop=True)
    E.assert_no_audio_leakage(us)
    return us


def stage_index():
    # vision: Open Images (human-verified image-level labels) + ImageNet class subsets
    oi = pd.read_csv(DATA / 'oi/oi_selection.csv')
    oi['path'] = oi.image_id.map(lambda i: str(DATA / f'oi/images/{i}.jpg'))
    oi = oi[oi.path.map(os.path.exists)]
    oi = oi.assign(source='openimages', cls=oi.kind)[['path', 'label', 'source', 'cls', 'tags']]
    inn = pd.read_csv(DATA / 'inet/inet_index.csv')
    inn['path'] = inn.file.map(lambda f: str(DATA / f'inet/images/{f}'))
    inn = inn.assign(source='imagenet', tags='')[['path', 'label', 'source', 'cls', 'tags']]
    vis = pd.concat([oi, inn], ignore_index=True)
    # drop unreadable / tiny images
    ok = []
    for p in vis.path:
        im = cv2.imread(p); ok.append(im is not None and min(im.shape[:2]) >= 64)
    vis = vis[np.array(ok)].reset_index(drop=True)
    vis = split_strat(vis, ['source', 'label'], (0.7, 0.15, 0.15), seed=42)
    vis.to_csv(WORK / 'vision_index.csv', index=False)
    log.info('vision %s', vis.groupby(['split', 'label']).size().unstack().to_dict())
    make_pairs(vis, load_us8k())


def stage_index_av():
    # VGGSound-EV: real, synchronized clips (frame at 5 s + audio 3-7 s)
    rows = []
    for f in sorted(glob.glob(str(DATA / 'vgg/out/shard_*.json'))):
        for k in json.load(open(f))['kept']:
            if k['ok']:
                rows.append(k)
    vgg = pd.DataFrame(rows)
    vgg['ytid'] = vgg.key.str[:-7]
    vgg['label'] = vgg['class'].isin(POS_VGG).astype(int)
    vgg['frame'] = vgg.key.map(lambda k: str(DATA / f'vgg/out/frames/{k}.jpg'))
    vgg['audio'] = vgg.key.map(lambda k: str(DATA / f'vgg/out/audio/{k}.wav'))
    # video-disjoint split: one split per YouTube id, stratified by the id's first class
    first = vgg.sort_values('key').groupby('ytid')['class'].first().reset_index()
    first = split_strat(first, ['class'], (0.6, 0.2, 0.2), seed=42)
    vgg['split'] = vgg.ytid.map(dict(zip(first.ytid, first.split)))
    vgg.to_csv(WORK / 'vgg_index.csv', index=False)
    log.info('vgg %s', vgg.groupby(['split', 'label']).size().unstack().to_dict())

    # audio branch training index: US8K (official folds) + VGGSound-EV audio of the same split
    us = load_us8k().assign(source='us8k')
    va = vgg.rename(columns={'audio': 'path', 'label': 'is_siren', 'ytid': 'source_id'})
    va = va.assign(classID=-1, fold=-1, source='vggsound')[['path', 'classID', 'is_siren', 'source_id', 'fold', 'split', 'source']]
    aud = pd.concat([us, va], ignore_index=True)
    aud.to_csv(WORK / 'audio_index.csv', index=False)
    log.info('audio %s', aud.groupby(['split', 'source', 'is_siren']).size().to_dict())

    # VGG leakage check (video-disjoint)
    ids = {s: set(g.ytid) for s, g in vgg.groupby('split')}
    assert not (ids['train'] & ids['test']) and not (ids['train'] & ids['val']) and not (ids['val'] & ids['test'])
    log.info('index_av done')


def make_pairs(v, us):
    # synthetic EV-AVSim-L pairs, 3 pairing seeds
    for seed in (42, 43, 44):
        out = {}
        for s, n in (('train', 20000), ('val', 4000), ('test', 4000)):
            pairs = E.build_pairs_for_split(s, v, us, n, seed=seed)
            out[s] = pd.DataFrame([p.__dict__ for p in pairs])
            out[s].to_csv(WORK / f'pairs_seed{seed}_{s}.csv', index=False)
        for a_, b_ in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            assert not set(out[a_].image_path) & set(out[b_].image_path)
            assert not set(out[a_].audio_path) & set(out[b_].audio_path)
        log.info('pairs seed %d: %s', seed, {s: (len(d), round(d.label.mean(), 3)) for s, d in out.items()})


# ----------------------------------------------------------------- training
def rand_crop(im, rng):
    h, w = im.shape[:2]
    for _ in range(10):
        area = h * w * rng.uniform(0.55, 1.0)
        ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        cw, ch = int(round(np.sqrt(area * ar))), int(round(np.sqrt(area / ar)))
        if cw <= w and ch <= h:
            x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
            im = im[y:y + ch, x:x + cw]; break
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        im = im[:, ::-1]
    return np.ascontiguousarray(im)


class ImgTrain(torch.utils.data.Dataset):
    def __init__(self, df, train, seed):
        self.p = df.path.tolist(); self.y = df.label.astype(float).tolist(); self.train = train
        self.rng = np.random.default_rng(seed)
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        im = cv2.imread(self.p[i])
        im = rand_crop(im, self.rng) if self.train else cv2.resize(im, (224, 224))
        return E.img_to_tensor(im), torch.tensor(self.y[i])


def fit(model, trl, val_fn, epochs, lr, tag, pos_weight=None, wd=1e-4, patience=4):
    from sklearn.metrics import average_precision_score
    ck = WORK / f'{tag}.pt'
    if ck.exists():
        model.load_state_dict(torch.load(ck)); log.info('loaded %s', ck); return model
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, hist, best_ep = -1, [], 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = 0; n = 0
        for x, y in trl:
            opt.zero_grad(); loss = crit(model(x).view(-1), y); loss.backward(); opt.step()
            tl += loss.item() * len(y); n += len(y)
        sch.step()
        p, t = val_fn(model)
        ap = average_precision_score(t, p)
        hist.append({'epoch': ep + 1, 'train_loss': tl / n, 'val_ap': ap})
        log.info('%s ep %d loss %.4f val AP %.4f (%.0fs)', tag, ep + 1, tl / n, ap, time.time() - t0)
        if ap > best:
            best = ap; best_ep = ep; torch.save(model.state_dict(), ck)
        elif ep - best_ep >= patience:
            log.info('%s early stop at epoch %d (best %d)', tag, ep + 1, best_ep + 1); break
    json.dump(hist, open(WORK / f'{tag}_history.json', 'w'), indent=1)
    model.load_state_dict(torch.load(ck))
    return model


@torch.no_grad()
def predict_images(model, paths, bs=64, corrupt=None, sev=0):
    model.eval(); out = []
    for i in range(0, len(paths), bs):
        ims = []
        for p in paths[i:i + bs]:
            im = E.load_img(p)
            if corrupt and sev:
                im = E.apply_visual(im, corrupt, sev, seed=sev)
            ims.append(E.img_to_tensor(im))
        out.append(torch.sigmoid(model(torch.stack(ims)).view(-1)).numpy())
    return np.concatenate(out)


def stage_vision(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    vis = pd.read_csv(WORK / 'vision_index.csv')
    tr, va = vis[vis.split == 'train'], vis[vis.split == 'val']
    trl = torch.utils.data.DataLoader(ImgTrain(tr, True, seed), batch_size=32, shuffle=True, num_workers=0)
    vp = va.path.tolist(); vy = va.label.to_numpy()
    m = E.build_vision()
    fit(m, trl, lambda mm: (predict_images(mm, vp), vy), int(os.environ.get('EPOCHS_V', 12)), 3e-4, f'vision_s{seed}')


def load_wave(p):
    import librosa
    y, _ = librosa.load(p, sr=SR, mono=True)
    n = int(4.0 * SR)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def mel_cache():
    f = WORK / 'mels.npz'
    aud = pd.read_csv(WORK / 'audio_index.csv')
    if f.exists():
        return aud, np.load(f)['m']
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(2) as ex:
        mels = list(ex.map(_mel_of, aud.path.tolist(), chunksize=64))
    M = np.stack(mels).astype(np.float16)
    np.savez(f, m=M)
    return aud, M


def _mel_of(p):
    return E.wav_to_mel(load_wave(p)).numpy()[0]


class MelDS(torch.utils.data.Dataset):
    def __init__(self, M, y, train, seed):
        self.M, self.y, self.train = M, y, train; self.rng = np.random.default_rng(seed)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        m = self.M[i].astype(np.float32).copy()
        if self.train:
            m = np.roll(m, self.rng.integers(-20, 21), axis=1)
            for _ in range(2):
                f = self.rng.integers(0, 9); f0 = self.rng.integers(0, 64 - f); m[f0:f0 + f] = m.min()
                t = self.rng.integers(0, 21); t0 = self.rng.integers(0, m.shape[1] - t); m[:, t0:t0 + t] = m.min()
        return torch.from_numpy(m).unsqueeze(0), torch.tensor(float(self.y[i]))


def stage_audio(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    aud, M = mel_cache()
    tr = (aud.split == 'train').to_numpy(); va = (aud.split == 'val').to_numpy()
    y = aud.is_siren.to_numpy()
    trl = torch.utils.data.DataLoader(MelDS(M[tr], y[tr], True, seed), batch_size=64, shuffle=True)
    vl = torch.utils.data.DataLoader(MelDS(M[va], y[va], False, seed), batch_size=256)
    pw = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), dtype=torch.float32)

    @torch.no_grad()
    def vf(m):
        m.eval(); P, T = [], []
        for x, t in vl:
            P.append(torch.sigmoid(m(x)).numpy()); T.append(t.numpy())
        return np.concatenate(P), np.concatenate(T)
    fit(E.AudioCNN(), trl, vf, int(os.environ.get('EPOCHS_A', 30)), 2e-3, f'audio_s{seed}', pos_weight=pw)


@torch.no_grad()
def predict_waves(model, waves, bs=128):
    model.eval(); out = []
    for i in range(0, len(waves), bs):
        x = torch.stack([E.wav_to_mel(w) for w in waves[i:i + bs]])
        out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out)


def models(seed):
    vm = E.build_vision(); vm.load_state_dict(torch.load(WORK / f'vision_s{seed}.pt')); vm.eval()
    am = E.AudioCNN(); am.load_state_dict(torch.load(WORK / f'audio_s{seed}.pt')); am.eval()
    return vm, am


# ----------------------------------------------------------------- scoring
VKEYS = E.VISUAL_FEATURES; AKEYS = E.ACOUSTIC_FEATURES


def score_sources(seed, image_paths, audio_paths, tag, vcorr=None, acorr=None, sev=0):
    """Per-source probabilities + descriptors. Returns dicts path -> (p, feat dict)."""
    f = WORK / f'src_{tag}_s{seed}.json'
    if f.exists():
        return json.load(open(f))
    vm, am = models(seed)
    ip = sorted(set(image_paths)); ap = sorted(set(audio_paths))
    pv = predict_images(vm, ip, corrupt=vcorr, sev=sev) if ip else []
    vf = {}
    for p in ip:
        im = E.load_img(p)
        if vcorr and sev:
            im = E.apply_visual(im, vcorr, sev, seed=sev)
        vf[p] = E.visual_features(im)
    waves = []
    for p in ap:
        w = load_wave(p)
        if acorr and sev:
            w = E.apply_acoustic(w, SR, acorr, sev, seed=sev)
        waves.append(w)
    pa = predict_waves(am, waves) if ap else []
    af = {p: E.acoustic_features(w, SR) for p, w in zip(ap, waves)}
    out = {'img': {p: [float(a), vf[p]] for p, a in zip(ip, pv)},
           'aud': {p: [float(a), af[p]] for p, a in zip(ap, pa)}}
    json.dump(out, open(f, 'w'))
    return out


def assemble(df, src, img_col, aud_col, label_col, scen_col):
    pv = np.array([src['img'][p][0] for p in df[img_col]])
    pa = np.array([src['aud'][p][0] for p in df[aud_col]])
    ft = np.vstack([E.build_feature_vector(src['img'][i][1], src['aud'][a][1], v, w)
                    for i, a, v, w in zip(df[img_col], df[aud_col], pv, pa)])
    ctx = [{"lighting": "night" if r[0] < 0.4 else "day", "weather": "fog" if r[3] > 0.5 else "clear",
            "traffic_density": "high" if r[5] > 0.12 else "low"} for r in ft]
    return E.SplitData(y=df[label_col].to_numpy().astype(int), p_vision_raw=pv, p_audio_raw=pa,
                       features=ft, scenario=df[scen_col].astype(str).to_numpy(), contexts=ctx)


def stage_score(seed):
    vgg = pd.read_csv(WORK / 'vgg_index.csv')
    ims, auds = set(), set()
    for ps in (42, 43, 44):
        for s in ('train', 'val', 'test'):
            d = pd.read_csv(WORK / f'pairs_seed{ps}_{s}.csv'); ims |= set(d.image_path); auds |= set(d.audio_path)
    score_sources(seed, sorted(ims), sorted(auds), 'synthetic')
    score_sources(seed, vgg.frame.tolist(), vgg.audio.tolist(), 'vgg')
    # vision branch on its own held-out image test split (natural images, incl. tagged conditions)
    vis = pd.read_csv(WORK / 'vision_index.csv')
    score_sources(seed, vis[vis.split == 'test'].path.tolist(), [], 'vistest')
    log.info('scored')


# ----------------------------------------------------------------- protocol
def fit_fusers(train, val, seed):
    cal_v = E.TemperatureScaler().fit(val.p_vision_raw, val.y)
    cal_a = E.TemperatureScaler().fit(val.p_audio_raw, val.y)
    cal = lambda d: (cal_v.transform(d.p_vision_raw), cal_a.transform(d.p_audio_raw))
    pv_tr, pa_tr = cal(train); pv_va, pa_va = cal(val)
    gate = E.ReliabilityGate(model='logreg', seed=seed).fit(train.features, pv_tr, pa_tr, train.y)
    mlp = E.ReliabilityGate(model='mlp', hidden=32, seed=seed).fit(train.features, pv_tr, pa_tr, train.y)
    from sklearn.linear_model import LogisticRegression
    late = LogisticRegression(max_iter=1000).fit(np.column_stack([pv_tr, pa_tr]), train.y)
    wv = E.grid_search_static_weight(pv_va, pa_va, val.y)

    def nll_bias(g):
        z = E.logit(pv_va); za = E.logit(pa_va); w = g.predict_weight(val.features)
        best = (np.inf, 0.0)
        for b in np.linspace(-4, 4, 161):
            s = np.clip(E.sigmoid(w * z + (1 - w) * za + b), 1e-6, 1 - 1e-6)
            nll = -np.mean(val.y * np.log(s) + (1 - val.y) * np.log(1 - s))
            best = min(best, (nll, float(b)))
        return best[1]
    F = dict(cal=cal, gate=gate, mlp=mlp, late=late, wv=wv, b=nll_bias(gate), b_mlp=nll_bias(mlp),
             T=(cal_v.temperature, cal_a.temperature))
    s_val = score_all(F, val)
    F['thr'] = {k: E.select_threshold(v, val.y, 'f1') for k, v in s_val.items()}
    F['ece'] = {'before': {'vision': E.expected_calibration_error(val.p_vision_raw, val.y),
                           'audio': E.expected_calibration_error(val.p_audio_raw, val.y)},
                'after': {'vision': E.expected_calibration_error(cal(val)[0], val.y),
                          'audio': E.expected_calibration_error(cal(val)[1], val.y)}}
    return F


def score_all(F, d):
    pv, pa = F['cal'](d)
    return {'vision_only': pv, 'audio_only': pa, 'mean': E.mean_fusion(pv, pa), 'max': E.max_fusion(pv, pa),
            'noisy_or': E.noisy_or(pv, pa), 'static': E.static_weighted(pv, pa, w_v=F['wv']),
            'learned_late': E.learned_late_fusion(pv, pa, model=F['late']),
            'rule_context': E.rule_based_context(pv, pa, contexts=d.contexts),
            'proposed': E.proposed_fusion(pv, pa, gate=F['gate'], features=d.features, bias=F['b']),
            'proposed_mlp': E.proposed_fusion(pv, pa, gate=F['mlp'], features=d.features, bias=F['b_mlp']),
            'oracle': E.oracle_gate(pv, pa, d.y)}


def boot(y, s, thr, B=1000, seed=0):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed); n = len(y); R = {k: [] for k in ('auprc', 'auroc', 'precision', 'recall', 'f1', 'recall_at_fpr10')}
    for _ in range(B):
        i = rng.integers(0, n, n); yy, ss = y[i], s[i]
        if yy.min() == yy.max():
            continue
        r = E.evaluate_scores(yy, ss, thr)
        for k in R:
            R[k].append(r[k])
    return {k: float(np.std(v)) for k, v in R.items()}


def evaluate(F, test, B=1000):
    S = score_all(F, test); res = {}
    for k, s in S.items():
        r = E.evaluate_scores(test.y, s, F['thr'][k]); r['std'] = boot(test.y, s, F['thr'][k], B)
        res[k] = r
    contenders = [k for k in res if k not in ('proposed', 'proposed_mlp', 'oracle')]
    best = max(contenders, key=lambda k: res[k]['auprc'])
    pred = lambda k: (S[k] >= F['thr'][k]).astype(int)
    sig = {g: {**E.mcnemar(test.y, pred(best), pred(g)), 'baseline': best} for g in ('proposed', 'proposed_mlp')}
    per = {}
    for k in ('vision_only', 'audio_only', 'noisy_or', 'max', 'static', 'proposed', 'proposed_mlp'):
        per[k] = {}
        for sc in sorted(set(test.scenario)):
            m = test.scenario == sc
            pr = pred(k)[m]; yy = test.y[m]
            per[k][sc] = {'n': int(m.sum()), 'acc': float((pr == yy).mean()),
                          'pos_rate_pred': float(pr.mean())}
    w = F['gate'].predict_weight(test.features)
    return S, {'test': res, 'sig': sig, 'per_scenario': per,
               'gate': {'mean_wv': float(w.mean()), 'std_wv': float(w.std()), 'coef': F['gate'].coefficients()},
               'T': F['T'], 'ece': F['ece'], 'static_wv': F['wv'], 'thr': F['thr']}


def stage_eval(seed):
    out = {}
    src = score_sources(seed, [], [], 'synthetic')
    for ps in (42, 43, 44):
        d = {s: assemble(pd.read_csv(WORK / f'pairs_seed{ps}_{s}.csv'), src, 'image_path', 'audio_path', 'label', 'scenario')
             for s in ('train', 'val', 'test')}
        F = fit_fusers(d['train'], d['val'], seed)
        _, r = evaluate(F, d['test'], B=1000 if ps == 42 else 200)
        out[f'synthetic_pair{ps}'] = r
        if ps == 42:
            F_syn = F
    # real VGGSound-EV
    vgg = pd.read_csv(WORK / 'vgg_index.csv'); vs = score_sources(seed, [], [], 'vgg')
    dv = {s: assemble(vgg[vgg.split == s], vs, 'frame', 'audio', 'label', 'class') for s in ('train', 'val', 'test')}
    Fv = fit_fusers(dv['train'], dv['val'], seed)
    S, r = evaluate(Fv, dv['test']); out['vgg_indomain'] = r
    # luminance-defined natural low light on real frames
    lum = dv['test'].features[:, 0]
    out['vgg_dark'] = {}
    for name, m in (('dark', lum < 0.25), ('bright', lum >= 0.25)):
        res = {}
        for k in ('vision_only', 'audio_only', 'noisy_or', 'max', 'static', 'proposed', 'proposed_mlp'):
            if len(set(dv['test'].y[m])) > 1:
                res[k] = E.evaluate_scores(dv['test'].y[m], S[k][m], Fv['thr'][k])
        out['vgg_dark'][name] = {'n': int(m.sum()), 'pos': int(dv['test'].y[m].sum()), 'res': res}
    # zero-shot: everything fitted on synthetic, applied to real test
    _, r0 = evaluate(F_syn, dv['test']); out['vgg_zeroshot'] = r0
    # vision branch on held-out natural images
    vis = pd.read_csv(WORK / 'vision_index.csv'); vt = vis[vis.split == 'test'].copy()
    vsrc = score_sources(seed, [], [], 'vistest'); vt['p'] = vt.path.map(lambda p: vsrc['img'][p][0])
    from sklearn.metrics import average_precision_score, roc_auc_score
    out['vision_test'] = {'n': len(vt), 'pos': int(vt.label.sum()),
                          'auprc': float(average_precision_score(vt.label, vt.p)), 'auroc': float(roc_auc_score(vt.label, vt.p)),
                          'by_source': {s: {'n': len(g), 'auprc': float(average_precision_score(g.label, g.p))} for s, g in vt.groupby('source')}}
    vt['lum'] = vt.path.map(lambda p: vsrc['img'][p][1]['luminance'])
    tagged = vt.tags.fillna('').str.contains('night|fog|rain|snow|darkness|mist|haze|dusk|storm|smoke')
    out['vision_test']['adverse_tagged_or_dark'] = {}
    for name, m in (('adverse', tagged | (vt.lum < 0.25)), ('other', ~(tagged | (vt.lum < 0.25)))):
        g = vt[m]
        out['vision_test']['adverse_tagged_or_dark'][name] = {'n': len(g), 'pos': int(g.label.sum()),
            'auprc': float(average_precision_score(g.label, g.p)) if g.label.nunique() > 1 else None,
            'auroc': float(roc_auc_score(g.label, g.p)) if g.label.nunique() > 1 else None}
    json.dump(out, open(WORK / f'results_s{seed}.json', 'w'), indent=1, default=float)
    log.info('eval done')


def stage_sweep(seed):
    f = WORK / f'sweep_s{seed}.json'
    src = score_sources(seed, [], [], 'synthetic')
    d = {s: assemble(pd.read_csv(WORK / f'pairs_seed42_{s}.csv'), src, 'image_path', 'audio_path', 'label', 'scenario')
         for s in ('train', 'val')}
    F = fit_fusers(d['train'], d['val'], seed)
    te = pd.read_csv(WORK / 'pairs_seed42_test.csv').iloc[:1500]
    res = json.load(open(f)) if f.exists() else {}
    names = {'vision_only': 'Vision only', 'audio_only': 'Audio only', 'static': 'Static weighted',
             'noisy_or': 'Noisy-OR', 'proposed': 'Proposed'}
    clean = assemble(te, src, 'image_path', 'audio_path', 'label', 'scenario')
    for ch, corr, sev in [('clean', None, 0)] + E.sweep_grid():
        key = f'{ch}|{corr}|{sev}'
        if key in res:
            continue
        if ch == 'clean':
            dd = clean
        else:
            s2 = score_sources(seed, te.image_path.tolist() if ch == 'visual' else [],
                               te.audio_path.tolist() if ch == 'acoustic' else [], f'sw_{corr}_{sev}',
                               vcorr=corr if ch == 'visual' else None, acorr=corr if ch == 'acoustic' else None, sev=sev)
            merged = {'img': {**src['img'], **s2['img']}, 'aud': {**src['aud'], **s2['aud']}}
            dd = assemble(te, merged, 'image_path', 'audio_path', 'label', 'scenario')
        S = score_all(F, dd)
        res[key] = {k: E.evaluate_scores(dd.y, S[k], F['thr'][k])['f1'] for k in names}
        res[key]['mean_wv'] = float(F['gate'].predict_weight(dd.features).mean())
        json.dump(res, open(f, 'w'), indent=1)
        log.info('sweep %s', key)
    log.info('sweep done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    {'index': lambda: stage_index(), 'index_av': lambda: stage_index_av(), 'vision': lambda: stage_vision(a.seed), 'audio': lambda: stage_audio(a.seed),
     'score': lambda: stage_score(a.seed), 'eval': lambda: stage_eval(a.seed), 'sweep': lambda: stage_sweep(a.seed)}[a.stage]()
