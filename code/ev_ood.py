"""
Out-of-distribution (unseen-source) experiment.

Training sources : ImageNet (images) + UrbanSound8K (audio) ONLY.
OOD test sources : Open Images photos + VGGSound audio/frames (never seen in training,
                   model selection, calibration or threshold choice).
Tests            : ID  = composed pairs, ImageNet test x UrbanSound8K fold 10
                   OOD-C = composed pairs, Open Images x VGGSound audio (same 5 scenarios)
                   OOD-R = real VGGSound-EV clips (frame + audio from the same clip)
Near-duplicates between training and OOD sources are detected (dHash for images,
log-Mel fingerprints for audio) and removed from the OOD sets.
usage: python ev_ood.py {index|vision|audio|score|eval} [--seed S]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_v3 as V
import evcore as E

MAIN = Path(os.environ.get('EV_WORK', 'work'))
OOD = Path(os.environ.get('EV_OOD', 'work_ood')); OOD.mkdir(exist_ok=True)
V.WORK = OOD   # models / caches of this experiment live here
log = V.log
VGG_SIREN = {'ambulance siren', 'fire truck siren', 'police car (siren)'}
VGG_CONF = {'vehicle horn, car horn, honking': 1, 'car engine idling': 5, 'civil defense siren': 9}


def dhash(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((im[:, 1:] > im[:, :-1]).flatten())


def hamming(A, B):
    # A: n x 8 uint8, B: m x 8 uint8 -> n x m distances
    out = np.zeros((len(A), len(B)), np.uint8)
    for k in range(8):
        x = np.bitwise_xor(A[:, k][:, None], B[:, k][None, :])
        out += np.unpackbits(x[..., None], axis=-1).sum(-1).astype(np.uint8)
    return out


def stage_index():
    vis = pd.read_csv(MAIN / 'vision_index.csv')
    vin = vis[vis.source == 'imagenet'].copy()          # keep the original IN splits
    voi = vis[vis.source == 'openimages'].copy()
    vgg = pd.read_csv(MAIN / 'vgg_index.csv')
    # ---- image near-duplicate check: training-side ImageNet vs Open Images and VGG frames
    hin = np.stack([dhash(p) for p in vin.path]); hoi = np.stack([dhash(p) for p in voi.path])
    hvf = np.stack([dhash(p) for p in vgg.frame])
    d_oi = hamming(hoi, hin).min(1); d_vf = hamming(hvf, hin).min(1)
    THR = 4
    voi['near_dup'] = d_oi <= THR; vgg['frame_near_dup'] = d_vf <= THR
    # ---- audio near-duplicate check: UrbanSound8K vs VGGSound audio (log-Mel fingerprints)
    aud = pd.read_csv(MAIN / 'audio_index.csv'); M = np.load(MAIN / 'mels.npz')['m'].astype(np.float32)
    def fp(m):
        x = m.reshape(len(m), 16, 4, 43, 4)[:, :, :, :43, :].mean((2, 4)) if m.shape[2] >= 172 else None
        x = x.reshape(len(x), -1); x = x - x.mean(1, keepdims=True)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    Mu = M[(aud.source == 'us8k').to_numpy()][:, :, :172]; Mv = M[(aud.source == 'vggsound').to_numpy()][:, :, :172]
    fu, fv = fp(Mu), fp(Mv)
    best = np.zeros(len(fv))
    for i in range(0, len(fv), 1000):
        best[i:i + 1000] = (fv[i:i + 1000] @ fu.T).max(1)
    vgg_paths = aud[aud.source == 'vggsound'].path.to_numpy()
    sim = dict(zip(vgg_paths, best))
    vgg['audio_max_sim_us8k'] = vgg.audio.map(sim)
    vgg['audio_near_dup'] = vgg.audio_max_sim_us8k >= 0.98
    dup = {'oi_images_near_dup': int(voi.near_dup.sum()), 'vgg_frames_near_dup': int(vgg.frame_near_dup.sum()),
           'vgg_audio_near_dup': int(vgg.audio_near_dup.sum()), 'image_hamming_threshold': THR,
           'audio_cosine_threshold': 0.98, 'vgg_audio_max_sim_quantiles': np.quantile(best, [0.5, 0.99, 1.0]).tolist()}
    json.dump(dup, open(OOD / 'duplicates.json', 'w'), indent=1)
    log.info('duplicates %s', dup)
    voi = voi[~voi.near_dup]; vgg = vgg[~(vgg.frame_near_dup | vgg.audio_near_dup)]
    vin.to_csv(OOD / 'vision_index.csv', index=False)
    voi.to_csv(OOD / 'ood_images.csv', index=False)
    vgg.to_csv(OOD / 'ood_vgg.csv', index=False)
    # audio branch: UrbanSound8K only
    keep = (aud.source == 'us8k').to_numpy()
    aud[keep].to_csv(OOD / 'audio_index.csv', index=False)
    np.savez(OOD / 'mels.npz', m=M[keep].astype(np.float16))
    # composed pairs. ID: ImageNet x UrbanSound8K (3 splits). OOD-C: Open Images x VGGSound audio.
    us = V.load_us8k()
    for s, n in (('train', 20000), ('val', 4000), ('test', 4000)):
        pairs = E.build_pairs_for_split(s, vin, us, n, seed=42)
        pd.DataFrame([p.__dict__ for p in pairs]).to_csv(OOD / f'pairs_id_{s}.csv', index=False)
    va = vgg.assign(path=vgg.audio, is_siren=vgg['class'].isin(VGG_SIREN).astype(int),
                    classID=vgg['class'].map(lambda c: 8 if c in VGG_SIREN else VGG_CONF.get(c, 0)), split='test')
    vo = voi.assign(split='test')
    pairs = E.build_pairs_for_split('test', vo, va, 4000, seed=42)
    df = pd.DataFrame([p.__dict__ for p in pairs])
    df['audio_class'] = df.audio_path.map(dict(zip(va.path, va['class'])))
    df.to_csv(OOD / 'pairs_oodc_test.csv', index=False)
    log.info('index: ID images %d, OOD images %d, OOD clips %d, OOD-C pairs %d (pos %.2f)',
             len(vin), len(voi), len(vgg), len(df), df.label.mean())


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def metrics(y, s, thr):
    r = E.evaluate_scores(y, s, thr)
    cm = np.array(r['confusion_matrix']); fp, tn = cm[0, 1], cm[0, 0]
    return {'auprc': r['auprc'], 'auroc': r['auroc'], 'f1': r['f1'], 'recall': r['recall'],
            'precision': r['precision'], 'fpr': float(fp / max(1, fp + tn)), 'ece': r['ece'],
            'brier': brier(y, s), 'recall_at_fpr10': r['recall_at_fpr10']}


def stage_score(seed):
    ims, auds = set(), set()
    for s in ('train', 'val', 'test'):
        d = pd.read_csv(OOD / f'pairs_id_{s}.csv'); ims |= set(d.image_path); auds |= set(d.audio_path)
    V.score_sources(seed, sorted(ims), sorted(auds), 'id')
    c = pd.read_csv(OOD / 'pairs_oodc_test.csv')
    V.score_sources(seed, sorted(set(c.image_path)), sorted(set(c.audio_path)), 'oodc')
    g = pd.read_csv(OOD / 'ood_vgg.csv')
    V.score_sources(seed, g.frame.tolist(), g.audio.tolist(), 'oodr')
    # branch-level OOD: all held-out Open Images photos, all VGGSound audio
    oi = pd.read_csv(OOD / 'ood_images.csv')
    V.score_sources(seed, oi.path.tolist(), [], 'oodimg')
    vis = pd.read_csv(OOD / 'vision_index.csv'); aud = pd.read_csv(OOD / 'audio_index.csv')
    V.score_sources(seed, vis[vis.split == 'test'].path.tolist(), aud[aud.split == 'test'].path.tolist(), 'idbranch')
    log.info('scored seed %d', seed)


def stage_eval(seed):
    sid = V.score_sources(seed, [], [], 'id')
    d = {k: V.assemble(pd.read_csv(OOD / f'pairs_id_{k}.csv'), sid, 'image_path', 'audio_path', 'label', 'scenario')
         for k in ('train', 'val', 'test')}
    F = V.fit_fusers(d['train'], d['val'], seed)          # everything fitted on in-distribution data
    out = {}
    sc = V.score_sources(seed, [], [], 'oodc')
    oodc = V.assemble(pd.read_csv(OOD / 'pairs_oodc_test.csv'), sc, 'image_path', 'audio_path', 'label', 'scenario')
    sr = V.score_sources(seed, [], [], 'oodr')
    g = pd.read_csv(OOD / 'ood_vgg.csv')
    oodr = V.assemble(g, sr, 'frame', 'audio', 'label', 'class')
    for name, data in (('id', d['test']), ('ood_composed', oodc), ('ood_real', oodr)):
        S = V.score_all(F, data)
        out[name] = {m: metrics(data.y, S[m], F['thr'][m]) for m in S}
        out[name]['n'] = int(len(data.y)); out[name]['pos'] = int(data.y.sum())
    # branch-level shift
    from sklearn.metrics import average_precision_score
    vis = pd.read_csv(OOD / 'vision_index.csv'); t = vis[vis.split == 'test']
    sb = V.score_sources(seed, [], [], 'idbranch')
    pid = np.array([sb['img'][p][0] for p in t.path])
    soi = V.score_sources(seed, [], [], 'oodimg'); oi = pd.read_csv(OOD / 'ood_images.csv')
    out['branch'] = {'vision_id_auprc': float(average_precision_score(t.label, pid)),
                     'vision_ood_auprc': float(average_precision_score(oi.label, [soi['img'][p][0] for p in oi.path])),
                     'n_ood_images': len(oi)}
    aud = pd.read_csv(OOD / 'audio_index.csv'); at = aud[aud.split == 'test']
    out['branch']['audio_id_auprc'] = float(average_precision_score(at.is_siren, [sb['aud'][p][0] for p in at.path]))
    out['branch']['audio_ood_auprc'] = float(average_precision_score(g.label, [sr['aud'][p][0] for p in g.audio]))
    out['T'] = F['T']; out['thr'] = F['thr']
    json.dump(out, open(OOD / f'results_ood_s{seed}.json', 'w'), indent=1, default=float)
    log.info('ood eval seed %d done', seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    {'index': stage_index, 'vision': lambda: V.stage_vision(a.seed), 'audio': lambda: V.stage_audio(a.seed),
     'score': lambda: stage_score(a.seed), 'eval': lambda: stage_eval(a.seed)}[a.stage]()
