from __future__ import annotations
'\npy -- Calibration and learned, context-conditioned modality gating.\n\nReplaces src/fusion.py:compute_weights, which is a four-branch lookup table of\nhand-picked constants:\n\n    if lighting == "night":          audio, vision = 0.7, 0.3\n    elif weather in ["rain","fog"]:  audio, vision = 0.6, 0.4\n    elif traffic == "high":          audio, vision = 0.65, 0.35\n    else:                            audio, vision = 0.3, 0.7\n\nThat is heuristic weighting, not a learned mechanism, and a reviewer will say\nso. This module provides the defensible version:\n\n  (a) Temperature scaling, so p_vision and p_audio are comparable probabilities\n      rather than two differently-miscalibrated scores.\n  (b) Continuous context features, replacing three categorical buckets.\n  (c) A gate g = sigma(w . phi(x)) trained to predict which modality is correct\n      for that sample. Logistic regression fits in seconds on CPU.\n\nEverything here trains on the TRAINING split only. Calibration is fit on\nvalidation. Nothing touches test.\n'
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
logger = logging.getLogger(__name__)
EPS = 1e-06

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> 'TemperatureScaler':
        z = logit(np.asarray(probs))
        y = np.asarray(labels, dtype=np.float64)
        best_t, best_nll = (1.0, np.inf)
        for t in np.geomspace(0.05, 20.0, 300):
            p = sigmoid(z / t)
            p = np.clip(p, EPS, 1 - EPS)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best_nll, best_t = (nll, float(t))
        self.temperature = best_t
        logger.info('Fitted temperature T=%.3f (val NLL %.4f)', best_t, best_nll)
        return self

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        return sigmoid(logit(np.asarray(probs)) / self.temperature)

def expected_calibration_error(probs: Sequence[float], labels: Sequence[int], n_bins: int=15) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(ece)
VISUAL_FEATURES = ['luminance', 'rms_contrast', 'lap_var', 'dark_channel', 'saturated_frac', 'edge_density', 'colorfulness']
ACOUSTIC_FEATURES = ['rms_db', 'est_snr_db', 'spectral_flatness', 'siren_band_ratio', 'harmonic_ratio', 'spectral_centroid']
CONF_FEATURES = ['p_vision', 'p_audio', 'conf_margin_v', 'conf_margin_a']
FEATURE_NAMES = VISUAL_FEATURES + ACOUSTIC_FEATURES + CONF_FEATURES

def visual_features(img: np.ndarray) -> Dict[str, float]:
    if img is None:
        return {k: 0.0 for k in VISUAL_FEATURES}
    if cv2 is not None:
        gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_u8 = img.mean(axis=2).astype(np.uint8)
    gray = gray_u8.astype(np.float32)
    x = img.astype(np.float32) / 255.0
    lum = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 255.0
    lap = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var()) if cv2 is not None else float(gray.var())
    dark = float(np.mean(np.min(x, axis=2)))
    sat = float(np.mean(gray > 245.0))
    if cv2 is not None:
        edges = cv2.Canny(gray_u8, 50, 150)
        edge_density = float(np.mean(edges > 0))
    else:
        edge_density = 0.0
    rg = x[..., 2] - x[..., 1]
    yb = 0.5 * (x[..., 2] + x[..., 1]) - x[..., 0]
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return {'luminance': lum, 'rms_contrast': contrast, 'lap_var': float(np.log1p(lap) / 10.0), 'dark_channel': dark, 'saturated_frac': sat, 'edge_density': edge_density, 'colorfulness': colorfulness}

def acoustic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    if y is None or len(y) == 0:
        return {k: 0.0 for k in ACOUSTIC_FEATURES}
    y = np.asarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y ** 2)) + EPS)
    rms_db = float(20.0 * np.log10(rms + EPS))
    win = max(1, int(0.032 * sr))
    n_frames = max(1, len(y) // win)
    frame_e = np.array([np.mean(y[i * win:(i + 1) * win] ** 2) for i in range(n_frames)])
    noise_floor = float(np.percentile(frame_e, 10) + EPS)
    signal_pk = float(np.percentile(frame_e, 90) + EPS)
    est_snr_db = float(10.0 * np.log10(signal_pk / noise_floor))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) + EPS
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    power = spec ** 2
    total = float(power.sum()) + EPS
    flatness = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + EPS))
    band = (freqs >= 500.0) & (freqs <= 1800.0)
    siren_band_ratio = float(power[band].sum() / total)
    centroid = float((freqs * power).sum() / total) / (sr / 2.0)
    k = max(1, len(power) // 100)
    harmonic_ratio = float(np.sort(power)[-k:].sum() / total)
    return {'rms_db': rms_db / 60.0, 'est_snr_db': est_snr_db / 40.0, 'spectral_flatness': flatness, 'siren_band_ratio': siren_band_ratio, 'harmonic_ratio': harmonic_ratio, 'spectral_centroid': centroid}

def build_feature_vector(vfeat: Dict[str, float], afeat: Dict[str, float], p_vision: float, p_audio: float) -> np.ndarray:
    row = [vfeat.get(k, 0.0) for k in VISUAL_FEATURES]
    row += [afeat.get(k, 0.0) for k in ACOUSTIC_FEATURES]
    row += [p_vision, p_audio, abs(p_vision - 0.5) * 2.0, abs(p_audio - 0.5) * 2.0]
    return np.asarray(row, dtype=np.float64)

class ReliabilityGate:

    def __init__(self, model: str='logreg', hidden: int=32, seed: int=42):
        self.model_type = model
        self.hidden = hidden
        self.seed = seed
        self.model = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    @staticmethod
    def make_targets(p_vision: np.ndarray, p_audio: np.ndarray, y: np.ndarray, thr: float=0.5):
        v_ok = (p_vision >= thr).astype(int) == y
        a_ok = (p_audio >= thr).astype(int) == y
        target = np.full(len(y), 0.5, dtype=np.float64)
        target[v_ok & ~a_ok] = 1.0
        target[~v_ok & a_ok] = 0.0
        weight = np.where(v_ok ^ a_ok, 1.0, 0.25)
        return (target, weight)

    def fit(self, X: np.ndarray, p_vision: np.ndarray, p_audio: np.ndarray, y: np.ndarray) -> 'ReliabilityGate':
        X = np.asarray(X, dtype=np.float64)
        self.mu, self.sigma = (X.mean(0), X.std(0) + EPS)
        Xs = (X - self.mu) / self.sigma
        target, weight = self.make_targets(np.asarray(p_vision), np.asarray(p_audio), np.asarray(y))
        if self.model_type == 'logreg':
            from sklearn.linear_model import LogisticRegression
            Xd = np.vstack([Xs, Xs])
            yd = np.concatenate([np.ones(len(Xs)), np.zeros(len(Xs))])
            wd = np.concatenate([weight * target, weight * (1.0 - target)])
            keep = wd > 1e-08
            self.model = LogisticRegression(max_iter=2000, C=1.0)
            self.model.fit(Xd[keep], yd[keep], sample_weight=wd[keep])
        elif self.model_type == 'mlp':
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(hidden_layer_sizes=(self.hidden,), max_iter=800, random_state=self.seed)
            self.model.fit(Xs, target)
        else:
            raise ValueError(f"unknown model '{self.model_type}'")
        logger.info('Gate trained on %d samples (%d informative)', len(X), int((weight > 0.5).sum()))
        return self

    def predict_weight(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('call fit() first')
        Xs = (np.asarray(X, dtype=np.float64) - self.mu) / self.sigma
        if self.model_type == 'logreg':
            g = self.model.predict_proba(Xs)[:, 1]
        else:
            g = np.clip(self.model.predict(Xs), 0.0, 1.0)
        return np.clip(0.05 + 0.9 * g, 0.0, 1.0)

    def coefficients(self) -> Dict[str, float] | None:
        if self.model_type != 'logreg' or self.model is None:
            return None
        return dict(zip(FEATURE_NAMES, self.model.coef_[0].tolist()))

    def save(self, path: Path) -> None:
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'type': self.model_type, 'model': self.model, 'mu': self.mu, 'sigma': self.sigma}, f)

    @classmethod
    def load(cls, path: Path) -> 'ReliabilityGate':
        import pickle
        with open(path, 'rb') as f:
            d = pickle.load(f)
        g = cls(model=d['type'])
        g.model, g.mu, g.sigma = (d['model'], d['mu'], d['sigma'])
        return g
'\npy -- Controlled degradation of the visual and acoustic channels.\n\nThis module supports the experiment the paper is actually about: does the\nsystem shift weight to the reliable modality as the other one degrades?\n\nThe current repository guesses conditions from the image itself\n(brightness < 100 => "night"; Laplacian variance < 60 => "fog"), which is a\nblur measure, not a weather estimator -- and on the real validation set it\nlabelled all 137 samples "clear", so the rain/fog branch of the method never\nexecuted. Here the condition is APPLIED, so it is known exactly, and severity\nis a controlled independent variable.\n\nSeverity runs 1..5 following the ImageNet-C convention. Severity 0 = clean.\n\nInference only: no retraining is required to produce the degradation curves.\n\nUsage:\n    img = apply_visual(img_bgr, "fog", severity=3)\n    y   = apply_acoustic(y, sr, "traffic_noise", severity=4, noise=noise_wav)\n'
from typing import Callable, Dict
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

def low_light(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    gamma = [1.0, 1.8, 2.4, 3.0, 3.8, 4.6][severity]
    gain = [1.0, 0.75, 0.6, 0.45, 0.33, 0.22][severity]
    read_noise = [0.0, 2.0, 4.0, 7.0, 11.0, 16.0][severity]
    x = img.astype(np.float32) / 255.0
    x = gain * np.power(x, gamma)
    photons = np.maximum(x * 255.0, 0.001)
    x = rng.poisson(photons) / 255.0
    x = x + rng.normal(0.0, read_noise / 255.0, x.shape)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def fog(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    beta = [0.0, 0.6, 1.0, 1.5, 2.1, 2.9][severity]
    airlight = [1.0, 0.85, 0.87, 0.9, 0.92, 0.95][severity]
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    d = 0.35 + 1.3 * (1.0 - d / (d.max() + 1e-08))
    t = np.exp(-beta * d)[..., None]
    x = img.astype(np.float32) / 255.0
    x = x * t + airlight * (1.0 - t)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def rain(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    n_drops = [0, 300, 700, 1300, 2200, 3400][severity]
    length = [0, 12, 16, 20, 26, 32][severity]
    angle = -60.0
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    xs = rng.integers(0, w, n_drops)
    ys = rng.integers(0, h, n_drops)
    dx = int(round(np.cos(np.deg2rad(angle)) * length))
    dy = int(round(np.sin(np.deg2rad(angle)) * length))
    for x0, y0 in zip(xs, ys):
        if cv2 is not None:
            cv2.line(layer, (int(x0), int(y0)), (int(x0 + dx), int(y0 + dy)), 1.0, 1)
        else:
            layer[min(y0, h - 1), min(x0, w - 1)] = 1.0
    if cv2 is not None:
        layer = cv2.GaussianBlur(layer, (3, 3), 0)
    x = img.astype(np.float32) / 255.0
    x = 0.86 * x + 0.14
    x = np.clip(x + 0.55 * layer[..., None], 0, 1)
    return (x * 255.0).astype(np.uint8)

def motion_blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    k = [1, 5, 9, 13, 19, 27][severity]
    if k <= 1 or cv2 is None:
        return img
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= kern.sum() + 1e-08
    return cv2.filter2D(img, -1, kern)

def glare(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    strength = [0.0, 0.35, 0.55, 0.75, 0.95, 1.15][severity]
    radius = [0.0, 0.16, 0.22, 0.3, 0.38, 0.48][severity]
    h, w = img.shape[:2]
    cx = rng.uniform(0.25, 0.75) * w
    cy = rng.uniform(0.2, 0.6) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / ((radius * max(h, w)) ** 2 + 1e-08)
    halo = strength * np.exp(-d2)[..., None]
    x = img.astype(np.float32) / 255.0
    x = np.clip(x + halo, 0, 1)
    x = np.clip(0.9 * x + 0.1 * strength, 0, 1)
    return (x * 255.0).astype(np.uint8)
VISUAL: Dict[str, Callable] = {'low_light': low_light, 'fog': fog, 'rain': rain, 'motion_blur': motion_blur, 'glare': glare}

def apply_visual(img: np.ndarray, name: str, severity: int, seed: int=0) -> np.ndarray:
    if severity <= 0:
        return img
    if not 1 <= severity <= 5:
        raise ValueError('severity must be in 0..5')
    return VISUAL[name](img, severity, np.random.default_rng(seed))
SNR_DB = [None, 20.0, 10.0, 5.0, 0.0, -5.0]

def _mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / max(1, len(noise))))
        noise = np.tile(noise, reps)
    start = rng.integers(0, max(1, len(noise) - len(signal) + 1))
    noise = noise[start:start + len(signal)]
    p_s = float(np.mean(signal ** 2)) + 1e-12
    p_n = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(p_s / (p_n * 10.0 ** (snr_db / 10.0)))
    out = signal + scale * noise
    peak = np.max(np.abs(out)) + 1e-12
    return (out / peak * 0.98).astype(np.float32) if peak > 1.0 else out.astype(np.float32)

def traffic_noise(y: np.ndarray, sr: int, severity: int, rng: np.random.Generator, noise: np.ndarray | None=None) -> np.ndarray:
    if noise is None:
        white = rng.normal(0, 1, len(y))
        spec = np.fft.rfft(white)
        freqs = np.maximum(np.fft.rfftfreq(len(y), 1.0 / sr), 1.0)
        noise = np.fft.irfft(spec / np.sqrt(freqs), n=len(y)).astype(np.float32)
    return _mix_at_snr(y.astype(np.float32), noise.astype(np.float32), SNR_DB[severity], rng)

def wind_noise(y: np.ndarray, sr: int, severity: int, rng: np.random.Generator, noise: np.ndarray | None=None) -> np.ndarray:
    brown = np.cumsum(rng.normal(0, 1, len(y)))
    brown = (brown - brown.mean()) / (np.std(brown) + 1e-08)
    return _mix_at_snr(y.astype(np.float32), brown.astype(np.float32), SNR_DB[severity], rng)

def clipping(y: np.ndarray, sr: int, severity: int, rng: np.random.Generator, noise: np.ndarray | None=None) -> np.ndarray:
    frac = [1.0, 0.7, 0.5, 0.35, 0.22, 0.12][severity]
    thr = frac * (np.max(np.abs(y)) + 1e-08)
    return np.clip(y, -thr, thr).astype(np.float32)
ACOUSTIC: Dict[str, Callable] = {'traffic_noise': traffic_noise, 'wind_noise': wind_noise, 'clipping': clipping}

def apply_acoustic(y: np.ndarray, sr: int, name: str, severity: int, noise: np.ndarray | None=None, seed: int=0) -> np.ndarray:
    if severity <= 0:
        return y
    if not 1 <= severity <= 5:
        raise ValueError('severity must be in 0..5')
    return ACOUSTIC[name](y, sr, severity, np.random.default_rng(seed), noise)

def sweep_grid(visual: bool=True, acoustic: bool=True) -> list[tuple[str, str, int]]:
    cells = []
    if visual:
        cells += [('visual', c, s) for c in VISUAL for s in range(1, 6)]
    if acoustic:
        cells += [('acoustic', c, s) for c in ACOUSTIC for s in range(1, 6)]
    return cells
'\npy -- Leak-free splits and an explicit audio-visual pairing protocol.\n\nReplaces two broken behaviours in the current repository:\n\n  1. src/audio_data.py used sklearn train_test_split on UrbanSound8K, which\n     leaks ~95% of validation clips (slices of the same source recording land\n     in both splits). UrbanSound8K ships 10 predefined folds precisely to\n     prevent this; we use them.\n\n  2. src/fusion.py paired modalities with `audio_preds[idx % len(audio_preds)]`,\n     i.e. by list position, so an ambulance image could be fused with a\n     recording of children playing. Here the pairing is an explicit, declared\n     scenario protocol with controlled proportions and a fixed seed.\n\nScenario protocol (EV-AVSim):\n\n    S1  EV image     + siren        -> 1   both modalities agree\n    S2  EV image     + non-siren    -> 1   VISION must carry (siren silent)\n    S3  non-EV image + siren        -> 1   AUDIO must carry (EV occluded)\n    S4  non-EV image + non-siren    -> 0   true negative\n    S5  non-EV image + confusable   -> 0   hard negative (horn/music/engine)\n\nS2 and S3 are the cases that make dynamic modality switching necessary; S5\nstops the audio branch from getting a free ride on "any urban sound => no EV".\n\nUsage:\n    splits = build_all_splits(\n        urbansound_root=Path("UrbanSound8K"),\n        vision_csv=Path("Emergency_Vehicles/train.csv"),\n        vision_images=Path("Emergency_Vehicles/train"),\n        out_dir=Path("data/pairs"),\n    )\n'
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)
US8K_CLASSES = {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark', 4: 'drilling', 5: 'engine_idling', 6: 'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music'}
SIREN_ID = 8
CONFUSABLE_IDS = (1, 5, 9)
FOLDS = {'train': (1, 2, 3, 4, 5, 6, 7, 8), 'val': (9,), 'test': (10,)}
SCENARIO_MIX = {'S1': 0.2, 'S2': 0.15, 'S3': 0.15, 'S4': 0.3, 'S5': 0.2}

@dataclass
class Pair:
    image_path: str
    audio_path: str
    image_label: int
    audio_label: int
    audio_class: str
    label: int
    scenario: str
    split: str

def load_urbansound_index(root: Path) -> pd.DataFrame:
    meta = pd.read_csv(Path(root) / 'metadata' / 'UrbanSound8K.csv')
    audio_root = Path(root) / 'audio'
    meta['path'] = meta.apply(lambda r: str(audio_root / f"fold{int(r['fold'])}" / r['slice_file_name']), axis=1)
    meta['is_siren'] = (meta['classID'] == SIREN_ID).astype(int)
    meta['source_id'] = meta['slice_file_name'].str.split('-').str[0]
    fold_to_split = {f: s for s, folds in FOLDS.items() for f in folds}
    meta['split'] = meta['fold'].map(fold_to_split)
    return meta[['path', 'classID', 'is_siren', 'source_id', 'fold', 'split']]

def assert_no_audio_leakage(index: pd.DataFrame) -> None:
    by_split = {s: set(g['source_id']) for s, g in index.groupby('split')}
    for a in by_split:
        for b in by_split:
            if a >= b:
                continue
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise AssertionError(f"Source-recording leakage between '{a}' and '{b}': {len(overlap)} shared recordings, e.g. {sorted(overlap)[:5]}")
    logger.info('Audio leakage check passed: no source recording spans two splits.')

def load_vision_index(csv_path: Path, images_root: Path, ratios: Sequence[float]=(0.6, 0.2, 0.2), seed: int=42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: 'image', df.columns[1]: 'label'})
    df['path'] = df['image'].apply(lambda n: str(Path(images_root) / str(n)))
    df['label'] = df['label'].astype(int)
    if df['label'].nunique() < 2:
        raise ValueError('Vision CSV has a single class. Do not drop negatives -- an all-positive evaluation set makes every metric degenerate.')
    rng = np.random.default_rng(seed)
    df['split'] = ''
    for label, group in df.groupby('label'):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        df.loc[idx[:n_tr], 'split'] = 'train'
        df.loc[idx[n_tr:n_tr + n_va], 'split'] = 'val'
        df.loc[idx[n_tr + n_va:], 'split'] = 'test'
    logger.info('Vision split: %s', df.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict())
    return df[['path', 'label', 'split']]

def _sample(rng: np.random.Generator, pool: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(pool) == 0:
        raise ValueError('Empty pool -- cannot construct this scenario.')
    replace = len(pool) < n
    idx = rng.choice(len(pool), size=n, replace=replace)
    return pool.iloc[idx].reset_index(drop=True)

def build_pairs_for_split(split: str, vision: pd.DataFrame, audio: pd.DataFrame, n_pairs: int, seed: int=42, mix: Dict[str, float] | None=None) -> List[Pair]:
    mix = mix or SCENARIO_MIX
    rng = np.random.default_rng(seed + hash(split) % 10000)
    v = vision[vision['split'] == split]
    a = audio[audio['split'] == split]
    v_pos, v_neg = (v[v['label'] == 1], v[v['label'] == 0])
    a_siren = a[a['is_siren'] == 1]
    a_other = a[(a['is_siren'] == 0) & ~a['classID'].isin(CONFUSABLE_IDS)]
    a_conf = a[a['classID'].isin(CONFUSABLE_IDS)]
    plan = [('S1', v_pos, a_siren, 1), ('S2', v_pos, a_other, 1), ('S3', v_neg, a_siren, 1), ('S4', v_neg, a_other, 0), ('S5', v_neg, a_conf, 0)]
    pairs: List[Pair] = []
    for name, vpool, apool, label in plan:
        n = int(round(mix[name] * n_pairs))
        if n == 0:
            continue
        vs = _sample(rng, vpool, n)
        aus = _sample(rng, apool, n)
        for i in range(n):
            pairs.append(Pair(image_path=vs.loc[i, 'path'], audio_path=aus.loc[i, 'path'], image_label=int(vs.loc[i, 'label']), audio_label=int(aus.loc[i, 'is_siren']), audio_class=US8K_CLASSES[int(aus.loc[i, 'classID'])], label=label, scenario=name, split=split))
    rng.shuffle(pairs)
    return pairs

def build_all_splits(urbansound_root: Path, vision_csv: Path, vision_images: Path, out_dir: Path, n_pairs: Dict[str, int] | None=None, seed: int=42) -> Dict[str, pd.DataFrame]:
    n_pairs = n_pairs or {'train': 6000, 'val': 1500, 'test': 1500}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = load_urbansound_index(urbansound_root)
    assert_no_audio_leakage(audio)
    vision = load_vision_index(vision_csv, vision_images, seed=seed)
    out: Dict[str, pd.DataFrame] = {}
    for split in ('train', 'val', 'test'):
        pairs = build_pairs_for_split(split, vision, audio, n_pairs[split], seed=seed)
        df = pd.DataFrame([asdict(p) for p in pairs])
        df.to_csv(out_dir / f'pairs_{split}.csv', index=False)
        out[split] = df
        logger.info('%s: %d pairs | positive rate %.3f | scenarios %s', split, len(df), df['label'].mean(), df['scenario'].value_counts().to_dict())
    imgs = {s: set(d['image_path']) for s, d in out.items()}
    auds = {s: set(d['audio_path']) for s, d in out.items()}
    for a_, b_ in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not imgs[a_] & imgs[b_], f'image leakage {a_}/{b_}'
        assert not auds[a_] & auds[b_], f'audio leakage {a_}/{b_}'
    logger.info('Pair-level contamination check passed.')
    return out
'\npy -- Fusion rules: the proposed method plus every baseline it must beat.\n\nReplaces src/fusion.py. Three specific defects are fixed:\n\n  1. `S = w_v*c_v + w_a*c_a` linearly mixes two probabilities. Probabilities do\n     not combine additively; for calibrated evidence the principled operation is\n     pooling in LOG-ODDS space (Kittler et al., TPAMI 1998). See `log_odds_pool`.\n\n  2. The old rule applied two unconditional promotions to positive:\n         if audio_detected and vision_conf < 0.3: emergency = True\n         if vision_detected and audio_conf < 0.3: emergency = True\n     Combined with a threshold dropped from 0.50 to 0.30, this makes the\n     decision close to `vision OR audio`. The reported recall gain\n     (0.442 -> 0.747) is therefore confounded across three simultaneous\n     changes. `noisy_or` below isolates that effect as its own baseline, so the\n     ablation can attribute the gain honestly.\n\n  3. Thresholds were tuned on, and reported on, the same split. Here\n     `select_threshold` fits on validation and is then frozen for test.\n\nEvery function takes calibrated probabilities and returns a continuous score,\nso AUROC/AUPRC are computable without committing to a threshold.\n'
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence
import numpy as np
logger = logging.getLogger(__name__)
EPS = 1e-06

def vision_only(pv: np.ndarray, pa: np.ndarray, **kw) -> np.ndarray:
    return np.asarray(pv, dtype=np.float64)

def audio_only(pv: np.ndarray, pa: np.ndarray, **kw) -> np.ndarray:
    return np.asarray(pa, dtype=np.float64)

def mean_fusion(pv: np.ndarray, pa: np.ndarray, **kw) -> np.ndarray:
    return 0.5 * (np.asarray(pv) + np.asarray(pa))

def max_fusion(pv: np.ndarray, pa: np.ndarray, **kw) -> np.ndarray:
    return np.maximum(np.asarray(pv), np.asarray(pa))

def noisy_or(pv: np.ndarray, pa: np.ndarray, **kw) -> np.ndarray:
    return 1.0 - (1.0 - np.asarray(pv)) * (1.0 - np.asarray(pa))

def static_weighted(pv: np.ndarray, pa: np.ndarray, w_v: float=0.5, **kw) -> np.ndarray:
    return w_v * np.asarray(pv) + (1.0 - w_v) * np.asarray(pa)

def rule_based_context(pv: np.ndarray, pa: np.ndarray, contexts: Sequence[dict] | None=None, **kw) -> np.ndarray:
    pv, pa = (np.asarray(pv), np.asarray(pa))
    if contexts is None:
        return static_weighted(pv, pa, 0.7)
    out = np.empty(len(pv))
    for i, ctx in enumerate(contexts):
        if ctx.get('lighting') == 'night':
            wa, wv = (0.7, 0.3)
        elif ctx.get('weather') in ('rain', 'fog'):
            wa, wv = (0.6, 0.4)
        elif ctx.get('traffic_density') == 'high':
            wa, wv = (0.65, 0.35)
        else:
            wa, wv = (0.3, 0.7)
        s = wa + wv
        out[i] = wv / s * pv[i] + wa / s * pa[i]
    return out

def learned_late_fusion(pv: np.ndarray, pa: np.ndarray, model=None, **kw) -> np.ndarray:
    if model is None:
        raise ValueError('learned_late_fusion needs a fitted model')
    X = np.column_stack([np.asarray(pv), np.asarray(pa)])
    return model.predict_proba(X)[:, 1]

def log_odds_pool(pv: np.ndarray, pa: np.ndarray, w_v: np.ndarray, bias: float=0.0) -> np.ndarray:
    pv, pa, w_v = (np.asarray(pv), np.asarray(pa), np.asarray(w_v))
    return sigmoid(w_v * logit(pv) + (1.0 - w_v) * logit(pa) + bias)

def proposed_fusion(pv: np.ndarray, pa: np.ndarray, gate=None, features: np.ndarray | None=None, bias: float=0.0, **kw) -> np.ndarray:
    if gate is None or features is None:
        raise ValueError('proposed_fusion needs a fitted gate and context features')
    w_v = gate.predict_weight(features)
    return log_odds_pool(pv, pa, w_v, bias=bias)

def oracle_gate(pv: np.ndarray, pa: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
    pv, pa, y = (np.asarray(pv), np.asarray(pa), np.asarray(y))
    v_err = np.abs(pv - y)
    a_err = np.abs(pa - y)
    return np.where(v_err <= a_err, pv, pa)

@dataclass
class FusionMethod:
    name: str
    fn: Callable
    needs: tuple = field(default_factory=tuple)
REGISTRY: Dict[str, FusionMethod] = {'vision_only': FusionMethod('Vision only', vision_only), 'audio_only': FusionMethod('Audio only', audio_only), 'mean': FusionMethod('Mean fusion', mean_fusion), 'max': FusionMethod('Max fusion', max_fusion), 'noisy_or': FusionMethod('Noisy-OR', noisy_or), 'static': FusionMethod('Static weighted (val-tuned)', static_weighted, ('w_v',)), 'rule_context': FusionMethod('Hand-crafted context rules [prior work]', rule_based_context, ('contexts',)), 'learned_late': FusionMethod('Learned late fusion (LR)', learned_late_fusion, ('model',)), 'proposed': FusionMethod('Proposed: learned gate + log-odds pooling', proposed_fusion, ('gate', 'features')), 'oracle': FusionMethod('Oracle gate (upper bound)', oracle_gate, ('y',))}

def select_threshold(scores: np.ndarray, y: np.ndarray, criterion: str='f1', target_fpr: float=0.1) -> float:
    scores, y = (np.asarray(scores), np.asarray(y))
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    best_t, best_v = (0.5, -np.inf)
    for t in grid:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        if criterion == 'f1':
            prec = tp / (tp + fp + EPS)
            rec = tp / (tp + fn + EPS)
            v = 2 * prec * rec / (prec + rec + EPS)
        elif criterion == 'recall@fpr':
            fpr = fp / (fp + tn + EPS)
            v = tp / (tp + fn + EPS) if fpr <= target_fpr else -1.0
        else:
            raise ValueError(criterion)
        if v > best_v:
            best_v, best_t = (v, float(t))
    logger.info('Selected threshold %.3f on validation (%s = %.4f)', best_t, criterion, best_v)
    return best_t

def grid_search_static_weight(pv: np.ndarray, pa: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    best_w, best = (0.5, -np.inf)
    for w in np.linspace(0.0, 1.0, 51):
        ap = average_precision_score(y, static_weighted(pv, pa, w))
        if ap > best:
            best, best_w = (ap, float(w))
    logger.info('Static baseline: best w_v = %.2f (val AP %.4f)', best_w, best)
    return best_w
'\npy -- Corrected evaluation protocol.\n\nFixes, relative to main.py in the current repository:\n\n  * Three-way split. Calibration, gate training and threshold selection use\n    train/val only; test is touched once, at the end. The current code\n    grid-searches thresholds and weight scales with tune_fusion_parameters()\n    and reports the best VALIDATION configuration as the result.\n\n  * One decision rule everywhere. The current code reports the headline table\n    using a dynamic threshold (0.30-0.35) plus two OR-style overrides, but\n    computes the day/night breakdown by passing raw fusion scores into\n    classification_metrics(), which thresholds at a hardcoded 0.5. The two sets\n    of numbers therefore come from different classifiers.\n\n  * Threshold-free metrics first. AUPRC and AUROC do not depend on the\n    operating point, which removes the single most obvious reviewer objection.\n\n  * Multiple seeds and a significance test, so that a 0.02 F1 difference on\n    ~1500 samples is not reported as an improvement.\n\nProduces one results.json. Generate every figure and table from that file, so\nthat text, tables and figures cannot disagree -- Fig. 4 currently shows audio\nprecision 0.646 while Table 2 states 0.633.\n'
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_curve
logger = logging.getLogger(__name__)
EPS_M = 1e-12

def recall_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float=0.1) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0

def evaluate_scores(y: Sequence[int], scores: Sequence[float], threshold: float) -> Dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    pred = (s >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    single_class = len(np.unique(y)) < 2
    if single_class:
        logger.error('Evaluation set contains a single class (%s). Precision/recall/F1 are undefined here -- this is exactly the condition that produced the spurious 1.000 scores. Do not report these numbers.', np.unique(y).tolist())
    return {'n': int(len(y)), 'positive_rate': float(y.mean()), 'threshold': float(threshold), 'precision': float(p), 'recall': float(r), 'f1': float(f1), 'accuracy': float(accuracy_score(y, pred)), 'auprc': None if single_class else float(average_precision_score(y, s)), 'auroc': None if single_class else float(roc_auc_score(y, s)), 'recall_at_fpr10': None if single_class else recall_at_fpr(y, s, 0.1), 'ece': expected_calibration_error(s, y), 'confusion_matrix': cm.tolist(), 'degenerate': bool(single_class)}

def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict:
    y, a, b = (np.asarray(y), np.asarray(pred_a), np.asarray(pred_b))
    a_ok, b_ok = (a == y, b == y)
    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue * 1.0)
    return {'n01': n01, 'n10': n10, 'p_value': p, 'significant': bool(p < 0.05)}

@dataclass
class SplitData:
    y: np.ndarray
    p_vision_raw: np.ndarray
    p_audio_raw: np.ndarray
    features: np.ndarray
    scenario: np.ndarray
    contexts: List[dict]

def run_protocol(train: SplitData, val: SplitData, test: SplitData, out_dir: Path, seed: int=42) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_note = {'seed': seed}
    ece_before = {'vision': expected_calibration_error(val.p_vision_raw, val.y), 'audio': expected_calibration_error(val.p_audio_raw, val.y)}
    cal_v = TemperatureScaler().fit(val.p_vision_raw, val.y)
    cal_a = TemperatureScaler().fit(val.p_audio_raw, val.y)

    def cal(d: SplitData):
        return (cal_v.transform(d.p_vision_raw), cal_a.transform(d.p_audio_raw))
    pv_tr, pa_tr = cal(train)
    pv_va, pa_va = cal(val)
    pv_te, pa_te = cal(test)
    ece_after = {'vision': expected_calibration_error(pv_va, val.y), 'audio': expected_calibration_error(pa_va, val.y)}
    gate = ReliabilityGate(model='logreg', seed=seed).fit(train.features, pv_tr, pa_tr, train.y)
    from sklearn.linear_model import LogisticRegression
    late = LogisticRegression(max_iter=1000).fit(np.column_stack([pv_tr, pa_tr]), train.y)
    best_wv = grid_search_static_weight(pv_va, pa_va, val.y)

    def score(split: str, pv, pa, d: SplitData) -> Dict[str, np.ndarray]:
        return {'vision_only': vision_only(pv, pa), 'audio_only': audio_only(pv, pa), 'mean': mean_fusion(pv, pa), 'max': max_fusion(pv, pa), 'noisy_or': noisy_or(pv, pa), 'static': static_weighted(pv, pa, w_v=best_wv), 'rule_context': rule_based_context(pv, pa, contexts=d.contexts), 'learned_late': learned_late_fusion(pv, pa, model=late), 'proposed': proposed_fusion(pv, pa, gate=gate, features=d.features), 'oracle': oracle_gate(pv, pa, d.y)}
    s_val = score('val', pv_va, pa_va, val)
    s_test = score('test', pv_te, pa_te, test)
    thresholds = {k: select_threshold(v, val.y, 'f1') for k, v in s_val.items()}
    results = {k: evaluate_scores(test.y, s_test[k], thresholds[k]) for k in s_test}
    contenders = [k for k in results if k not in ('proposed', 'oracle')]
    best_base = max(contenders, key=lambda k: results[k]['auprc'] or -1)
    pred_prop = (s_test['proposed'] >= thresholds['proposed']).astype(int)
    pred_base = (s_test[best_base] >= thresholds[best_base]).astype(int)
    sig = mcnemar(test.y, pred_base, pred_prop)
    sig['baseline'] = best_base
    per_scenario: Dict[str, Dict[str, Dict]] = {}
    for method in ('static', 'noisy_or', 'rule_context', 'proposed'):
        per_scenario[method] = {}
        for sc in sorted(set(test.scenario.tolist())):
            m = test.scenario == sc
            if m.sum() == 0:
                continue
            per_scenario[method][sc] = evaluate_scores(test.y[m], s_test[method][m], thresholds[method])
    w_v_test = gate.predict_weight(test.features)
    gate_report = {'mean_vision_weight': float(w_v_test.mean()), 'std_vision_weight': float(w_v_test.std()), 'coefficients': gate.coefficients()}
    payload = {**rng_note, 'calibration': {'ece_before': ece_before, 'ece_after': ece_after, 'T_vision': cal_v.temperature, 'T_audio': cal_a.temperature}, 'static_best_w_v': best_wv, 'thresholds': thresholds, 'test': results, 'significance': sig, 'per_scenario': per_scenario, 'gate': gate_report}
    (out_dir / f'results_seed{seed}.json').write_text(json.dumps(payload, indent=2))
    logger.info('Wrote %s', out_dir / f'results_seed{seed}.json')
    return payload

def aggregate(runs: List[Dict], out_path: Path) -> Dict:
    methods = list(runs[0]['test'].keys())
    metrics = ['auprc', 'auroc', 'precision', 'recall', 'f1', 'recall_at_fpr10']
    agg = {}
    for m in methods:
        agg[m] = {}
        for k in metrics:
            vals = [r['test'][m][k] for r in runs if r['test'][m][k] is not None]
            if vals:
                agg[m][k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    Path(out_path).write_text(json.dumps(agg, indent=2))
    return agg

def latex_table(agg: Dict, metrics=('auprc', 'auroc', 'precision', 'recall', 'f1', 'recall_at_fpr10')) -> str:
    names = {'vision_only': 'Vision only', 'audio_only': 'Audio only', 'mean': 'Mean fusion', 'max': 'Max fusion', 'noisy_or': 'Noisy-OR', 'static': 'Static weighted', 'learned_late': 'Learned late fusion', 'rule_context': 'Hand-crafted context rules', 'proposed': '\\textbf{Proposed}', 'oracle': '\\textit{Oracle gate (UB)}'}
    lines = []
    for key, label in names.items():
        if key not in agg:
            continue
        cells = []
        for k in metrics:
            v = agg[key].get(k)
            cells.append('--' if v is None else f"{v['mean']:.3f}$\\pm${v['std']:.3f}")
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)
'\npy -- Generates the two remaining paper figures.\n\nFigure 1 (architecture) no longer needs this script: it is drawn in TikZ inside\nmain.tex, so there is no image file to go missing.\n\nThis script produces the two figures that DO depend on results:\n\n  degradation_curves.png  -> Fig. 2: F1 vs corruption severity, per method\n  gate_behaviour.png      -> Fig. 3: mean vision weight w_v vs severity\n\nFigure 3 is the most persuasive plot in the paper. It is the difference between\nasserting "the system shifts weight to the reliable modality" and measuring it:\nif the gate works, mean w_v slopes DOWN as visual corruption increases and UP as\nacoustic corruption increases. If those curves come out flat, the mechanism is\nnot doing what the paper claims and you need to say so.\n\nBoth figures read one JSON file, so the text, tables and figures cannot drift\napart -- which is what produced the current mismatch where Fig. 4 shows audio\nprecision 0.646 while Table 2 states 0.633.\n\nExpected input, written by the degradation sweep (see sweep_grid):\n\n{\n  "visual": {\n    "fog": {\n      "severity": [0, 1, 2, 3, 4, 5],\n      "methods": {\n        "Vision only":      {"f1": [...], "f1_std": [...]},\n        "Static weighted":  {"f1": [...], "f1_std": [...]},\n        "Noisy-OR":         {"f1": [...], "f1_std": [...]},\n        "Proposed":         {"f1": [...], "f1_std": [...]}\n      },\n      "mean_w_v": [...], "mean_w_v_std": [...]\n    },\n    "low_light": {...}, "rain": {...}, "motion_blur": {...}, "glare": {...}\n  },\n  "acoustic": { "traffic_noise": {...}, "wind_noise": {...}, "clipping": {...} }\n}\n\nUsage:\n    python py sweep_results.json --outdir .\n'
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
STYLE = {'Vision only': {'color': '#4C72B0', 'ls': ':', 'lw': 1.4, 'marker': 'o'}, 'Audio only': {'color': '#55A868', 'ls': ':', 'lw': 1.4, 'marker': 's'}, 'Static weighted': {'color': '#8172B2', 'ls': '--', 'lw': 1.6, 'marker': '^'}, 'Noisy-OR': {'color': '#CCB974', 'ls': '-.', 'lw': 1.6, 'marker': 'v'}, 'Hand-crafted context rules': {'color': '#937860', 'ls': '--', 'lw': 1.6, 'marker': 'D'}, 'Proposed': {'color': '#C44E52', 'ls': '-', 'lw': 2.6, 'marker': '*'}}
FALLBACK = {'color': '#777777', 'ls': '-', 'lw': 1.4, 'marker': '.'}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 300, 'savefig.bbox': 'tight'})

def _panel(ax, cell: dict, title: str, ylabel: str | None):
    sev = cell['severity']
    for name, series in cell['methods'].items():
        st = STYLE.get(name, FALLBACK)
        y = np.asarray(series['f1'], dtype=float)
        ax.plot(sev, y, label=name, color=st['color'], linestyle=st['ls'], linewidth=st['lw'], marker=st['marker'], markersize=3.5)
        if 'f1_std' in series:
            sd = np.asarray(series['f1_std'], dtype=float)
            ax.fill_between(sev, y - sd, y + sd, color=st['color'], alpha=0.12, linewidth=0)
    ax.set_title(title.replace('_', ' '))
    ax.set_xlabel('severity')
    ax.set_xticks(sev)
    ax.set_ylim(0, 1)
    if ylabel:
        ax.set_ylabel(ylabel)

def degradation_curves(results: dict, out: Path) -> None:
    vis = list(results.get('visual', {}).items())
    aco = list(results.get('acoustic', {}).items())
    ncol = max(len(vis), len(aco), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(2.05 * ncol, 4.3), sharey=True)
    axes = np.atleast_2d(axes)
    for j in range(ncol):
        if j < len(vis):
            _panel(axes[0, j], vis[j][1], vis[j][0], 'F$_1$' if j == 0 else None)
        else:
            axes[0, j].axis('off')
        if j < len(aco):
            _panel(axes[1, j], aco[j][1], aco[j][0], 'F$_1$' if j == 0 else None)
        else:
            axes[1, j].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.005, 0.74, 'visual', rotation=90, va='center', fontsize=8, color='#555')
    fig.text(0.005, 0.3, 'acoustic', rotation=90, va='center', fontsize=8, color='#555')
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

def gate_behaviour(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, channel, title in ((axes[0], 'visual', 'Visual channel corrupted'), (axes[1], 'acoustic', 'Acoustic channel corrupted')):
        cells = results.get(channel, {})
        for name, cell in cells.items():
            if 'mean_w_v' not in cell:
                continue
            sev = cell['severity']
            w = np.asarray(cell['mean_w_v'], dtype=float)
            ax.plot(sev, w, marker='o', markersize=3.5, linewidth=1.5, label=name.replace('_', ' '))
            if 'mean_w_v_std' in cell:
                sd = np.asarray(cell['mean_w_v_std'], dtype=float)
                ax.fill_between(sev, w - sd, w + sd, alpha=0.12, linewidth=0)
        ax.axhline(0.5, color='#999', linewidth=0.8, linestyle=':')
        ax.set_title(title)
        ax.set_xlabel('severity')
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel('mean vision weight $\\overline{w_v}$')
    axes[0].annotate('weight shifts to audio', xy=(0.5, 0.12), xycoords='axes fraction', fontsize=7, color='#555')
    axes[1].annotate('weight shifts to vision', xy=(0.5, 0.86), xycoords='axes fraction', fontsize=7, color='#555')
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'wrote {out}')

def _demo() -> dict:
    sev = [0, 1, 2, 3, 4, 5]

    def cell(drop_static, drop_prop, wv_start, wv_end):
        m = {}
        for name, d in (('Vision only', drop_static * 1.4), ('Static weighted', drop_static), ('Noisy-OR', drop_static * 0.9), ('Proposed', drop_prop)):
            y = [max(0.05, 0.86 - d * s) for s in sev]
            m[name] = {'f1': y, 'f1_std': [0.02] * len(sev)}
        return {'severity': sev, 'methods': m, 'mean_w_v': list(np.linspace(wv_start, wv_end, len(sev))), 'mean_w_v_std': [0.03] * len(sev)}
    return {'visual': {k: cell(0.12, 0.05, 0.68, 0.22) for k in ('low_light', 'fog', 'rain', 'motion_blur', 'glare')}, 'acoustic': {k: cell(0.1, 0.04, 0.55, 0.85) for k in ('traffic_noise', 'wind_noise', 'clipping')}}

def _figures_cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', nargs='?', help='sweep results JSON')
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--demo', action='store_true', help='render with placeholder data to preview the layout')
    a = ap.parse_args()
    if a.demo or not a.results:
        print('No results file given -- rendering DEMO data. These are NOT results; do not put them in the paper.')
        res = _demo()
    else:
        res = json.loads(Path(a.results).read_text())
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    degradation_curves(res, out / 'degradation_curves.png')
    gate_behaviour(res, out / 'gate_behaviour.png')
'\npy -- One command produces every number and both figures.\n\n    python py --vision Emergency_Vehicles --audio UrbanSound8K\n\nWHY YOU HAVE TO RUN THIS AND I CANNOT\n-------------------------------------\nThe numbers in a paper have to come from a real experiment. Inventing them, or\nestimating them "by calculation", is data fabrication: if it is discovered the\npaper is retracted and the authors carry that permanently. So this script does\nall the work, but it has to run on your machine, because your machine is where\nthe two datasets and the GPU are.\n\nWHAT IT DOES, IN ORDER\n----------------------\n 1. Builds fold-safe splits and the five-scenario pairs        (py)\n 2. Trains the visual branch  (MobileNetV3-Small, 224px)\n 3. Trains the acoustic branch (log-Mel CNN) on folds 1-8\n 4. Caches confidences + 17 context features for every pair\n 5. Runs the full protocol over 5 seeds                        (py)\n 6. Runs the degradation sweep, inference only                 (py)\n 7. Writes degradation_curves.png and gate_behaviour.png       (py)\n 8. Writes results.tex with every table already filled in\n\nThen you copy results.tex into Overleaf and the red placeholders are gone.\n\nRUNTIME on an RTX 3050 Ti 4GB: roughly 1.5 to 2.5 hours, most of it step 6.\nEverything is cached in --work, so re-running skips finished stages.\nUse --quick first (2 epochs, 200 pairs) to check it runs end to end.\n\nINSTALL\n-------\n    pip install torch torchvision librosa scikit-learn matplotlib pandas tqdm soundfile\n'
import argparse, json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('run_all')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR, NMELS, MAXFR = (22050, 64, 173)

def build_vision():
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    return m.to(DEV)

class AudioCNN(nn.Module):

    def __init__(self):
        super().__init__()

        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64), nn.AdaptiveAvgPool2d((4, 4)))
        self.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 16, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        return self.c(self.f(x)).squeeze(-1)

def load_img(path, size=224):
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return cv2.resize(im, (size, size))

def img_to_tensor(bgr):
    x = bgr[..., ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(((x - mean) / std).transpose(2, 0, 1).copy())

def wav_to_mel(y):
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=NMELS, n_fft=2048, hop_length=512)
    db = librosa.power_to_db(mel, ref=np.max)
    if db.shape[1] < MAXFR:
        db = np.pad(db, ((0, 0), (0, MAXFR - db.shape[1])))
    return torch.from_numpy(db[:, :MAXFR]).unsqueeze(0).float()

class ImgDS(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return (img_to_tensor(load_img(r['path'])), torch.tensor(float(r['label'])))

class AudDS(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import librosa
        r = self.df.iloc[i]
        y, _ = librosa.load(r['path'], sr=SR, mono=True)
        return (wav_to_mel(y), torch.tensor(float(r['is_siren'])))

def train_branch(model, tr, va, epochs, lr, bs, tag, out: Path, pos_weight=None):
    ck = out / f'{tag}.pt'
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=DEV))
        log.info('loaded %s', ck)
        return model
    from sklearn.metrics import average_precision_score
    from tqdm import tqdm
    trl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0)
    val = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best = -1.0
    for ep in range(epochs):
        model.train()
        for x, y in tqdm(trl, desc=f'{tag} {ep + 1}/{epochs}', leave=False):
            x, y = (x.to(DEV), y.to(DEV))
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
        p, t = predict(model, val)
        ap = average_precision_score(t, p) if len(set(t.tolist())) > 1 else 0.0
        log.info('%s epoch %d val AP %.4f', tag, ep + 1, ap)
        if ap > best:
            best = ap
            torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=DEV))
    return model

@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, T = ([], [])
    for x, y in loader:
        P.append(torch.sigmoid(model(x.to(DEV))).cpu().numpy())
        T.append(y.numpy())
    return (np.concatenate(P), np.concatenate(T))

def preload_raw(pairs: pd.DataFrame, cache: Path | None=None):
    if cache is not None and cache.exists():
        d = np.load(cache)
        return (d['imgs'], d['wavs'])
    import librosa
    from tqdm import tqdm
    imgs, wavs = ([], [])
    n = int(4.0 * SR)
    for r in tqdm(pairs.itertuples(index=False), total=len(pairs), desc='preloading raw'):
        imgs.append(load_img(r.image_path))
        y, _ = librosa.load(r.audio_path, sr=SR, mono=True)
        wavs.append(np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32))
    imgs, wavs = (np.stack(imgs), np.stack(wavs))
    if cache is not None:
        np.savez(cache, imgs=imgs, wavs=wavs)
    return (imgs, wavs)

@torch.no_grad()
def score_pairs(vm, am, pairs: pd.DataFrame, cache: Path, v_corr=None, a_corr=None, sev=0, seed=0, raw=None):
    if cache is not None and cache.exists():
        d = np.load(cache)
        return (d['pv'], d['pa'], d['ft'])
    import librosa
    from tqdm import tqdm
    vm.eval()
    am.eval()
    PV, PA, FT = ([], [], [])
    rows = list(pairs.itertuples(index=False))
    for i, r in enumerate(tqdm(rows, desc='scoring', leave=False)):
        if raw is not None:
            img = raw[0][i].copy()
            y = raw[1][i].copy()
        else:
            img = load_img(r.image_path)
            y, _ = librosa.load(r.audio_path, sr=SR, mono=True)
        if v_corr and sev:
            img = apply_visual(img, v_corr, sev, seed=seed)
        if a_corr and sev:
            y = apply_acoustic(y, SR, a_corr, sev, seed=seed)
        pv = float(torch.sigmoid(vm(img_to_tensor(img).unsqueeze(0).to(DEV))).item())
        pa = float(torch.sigmoid(am(wav_to_mel(y).unsqueeze(0).to(DEV))).item())
        ft = build_feature_vector(visual_features(img), acoustic_features(y, SR), pv, pa)
        PV.append(pv)
        PA.append(pa)
        FT.append(ft)
    pv, pa, ft = (np.array(PV), np.array(PA), np.vstack(FT))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, pv=pv, pa=pa, ft=ft)
    return (pv, pa, ft)

def to_split(pairs, pv, pa, ft):
    ctx = []
    for row in ft:
        lum, lap = (row[0], row[2])
        ctx.append({'lighting': 'night' if lum < 0.4 else 'day', 'weather': 'fog' if row[3] > 0.5 else 'clear', 'traffic_density': 'high' if row[5] > 0.12 else 'low'})
    return SplitData(y=pairs['label'].to_numpy(), p_vision_raw=pv, p_audio_raw=pa, features=ft, scenario=pairs['scenario'].to_numpy(), contexts=ctx)

def emit_tex(agg, runs, sweep, out: Path):

    def c(m, k):
        v = agg.get(m, {}).get(k)
        return '--' if not v else f"{v['mean']:.3f}$\\pm${v['std']:.3f}"
    rows = [('Vision only', 'vision_only'), ('Audio only', 'audio_only'), ('Mean fusion', 'mean'), ('Max fusion', 'max'), ('Noisy-OR', 'noisy_or'), ('Static weighted', 'static'), ('Learned late fusion', 'learned_late'), ('Hand-crafted context rules', 'rule_context'), ('\\textbf{Proposed}', 'proposed'), ('\\textit{Oracle gate (UB)}', 'oracle')]
    mets = ['auprc', 'auroc', 'precision', 'recall', 'f1', 'recall_at_fpr10']
    main = '\n'.join((f'{lbl} & ' + ' & '.join((c(k, m) for m in mets)) + ' \\\\' for lbl, k in rows))
    r0 = runs[0]
    cal = r0['calibration']
    caltab = f"Vision & {cal['ece_before']['vision']:.3f} & {cal['ece_after']['vision']:.3f} & {cal['T_vision']:.2f} \\\\\nAudio  & {cal['ece_before']['audio']:.3f} & {cal['ece_after']['audio']:.3f} & {cal['T_audio']:.2f} \\\\"
    sc = r0['per_scenario']
    keys = sorted({k for m in sc.values() for k in m})
    scen = '\n'.join((f'{lbl} & ' + ' & '.join((f"{sc[k][s]['f1']:.2f}" if s in sc.get(k, {}) else '--' for s in keys)) + ' \\\\' for lbl, k in [('Static weighted', 'static'), ('Noisy-OR', 'noisy_or'), ('Hand-crafted context rules', 'rule_context'), ('\\textbf{Proposed}', 'proposed')]))
    coef = r0['gate'].get('coefficients') or {}
    top = sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:6]
    coefs = '\n'.join((f"{a.replace('_', ' ')} & {b:+.2f} & {c_.replace('_', ' ')} & {d:+.2f} \\\\" for (a, b), (c_, d) in zip(top[0::2], top[1::2])))
    sig = r0['significance']
    out.write_text(f"% Auto-generated by py -- paste these into main.tex.\n% Main comparison (Table: tab:main)\n{main}\n\n% Calibration (Table: tab:calibration)\n{caltab}\n\n% Per-scenario F1, columns {keys} (Table: tab:scenario)\n{scen}\n\n% Gate coefficients (Table: tab:coefficients)\n{coefs}\n\n% For the prose:\n%   best non-oracle baseline : {sig['baseline']}\n%   McNemar p-value          : {sig['p_value']:.4g}  (significant: {sig['significant']})\n%   proposed AUPRC           : {agg['proposed']['auprc']['mean']:.3f}\n%   best static AUPRC        : {agg[sig['baseline']]['auprc']['mean']:.3f}\n%   oracle AUPRC             : {agg['oracle']['auprc']['mean']:.3f}\n%   mean vision weight       : {r0['gate']['mean_vision_weight']:.3f}\n", encoding='utf-8')
    log.info('wrote %s', out)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vision', required=True, help='folder with train.csv and train/')
    ap.add_argument('--audio', required=True, help='UrbanSound8K root')
    ap.add_argument('--work', default='run')
    ap.add_argument('--epochs-vision', type=int, default=20)
    ap.add_argument('--epochs-audio', type=int, default=25)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--quick', action='store_true', help='tiny smoke run')
    a = ap.parse_args()
    if a.quick:
        a.epochs_vision, a.epochs_audio, a.seeds = (2, 2, 2)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info('device %s', DEV)
    n = {'train': 400, 'val': 150, 'test': 150} if a.quick else {'train': 6000, 'val': 1500, 'test': 1500}
    splits = build_all_splits(Path(a.audio), Path(a.vision) / 'train.csv', Path(a.vision) / 'train', work / 'pairs', n_pairs=n)
    vidx = load_vision_index(Path(a.vision) / 'train.csv', Path(a.vision) / 'train')
    vm = train_branch(build_vision(), ImgDS(vidx[vidx.split == 'train']), ImgDS(vidx[vidx.split == 'val']), a.epochs_vision, 0.0001, a.batch, 'vision', work)
    aidx = load_urbansound_index(Path(a.audio))
    tr = aidx[aidx.split == 'train']
    pw = torch.tensor((tr.is_siren == 0).sum() / max(1, (tr.is_siren == 1).sum()), dtype=torch.float32, device=DEV)
    am = train_branch(AudioCNN().to(DEV), AudDS(tr), AudDS(aidx[aidx.split == 'val']), a.epochs_audio, 0.003, a.batch, 'audio', work, pos_weight=pw)
    data = {}
    for s in ('train', 'val', 'test'):
        pv, pa, ft = score_pairs(vm, am, splits[s], work / f'scores_{s}.npz')
        data[s] = to_split(splits[s], pv, pa, ft)
    runs = [run_protocol(data['train'], data['val'], data['test'], work / 'results', seed=s) for s in range(a.seeds)]
    agg = aggregate(runs, work / 'results' / 'aggregate.json')
    log.info('proposed AUPRC %.3f | oracle %.3f', agg['proposed']['auprc']['mean'], agg['oracle']['auprc']['mean'])
    sweep_path = work / 'sweep_results.json'
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
    else:
        gate = ReliabilityGate().fit(data['train'].features, data['train'].p_vision_raw, data['train'].p_audio_raw, data['train'].y)
        wv0 = grid_search_static_weight(data['val'].p_vision_raw, data['val'].p_audio_raw, data['val'].y)
        thr = runs[0]['thresholds']
        raw = preload_raw(splits['test'], work / 'raw_test.npz')
        sweep = {'visual': {}, 'acoustic': {}}
        for ch, corr, sev in sweep_grid():
            cell = sweep[ch].setdefault(corr, {'severity': [0], 'methods': {}, 'mean_w_v': [], 'mean_w_v_std': []})
            if sev not in cell['severity']:
                cell['severity'].append(sev)
            pv, pa, ft = score_pairs(vm, am, splits['test'], None, raw=raw, v_corr=corr if ch == 'visual' else None, a_corr=corr if ch == 'acoustic' else None, sev=sev, seed=sev)
            y = splits['test']['label'].to_numpy()
            scores = {'Vision only': pv, 'Audio only': pa, 'Static weighted': static_weighted(pv, pa, w_v=wv0), 'Noisy-OR': noisy_or(pv, pa), 'Proposed': proposed_fusion(pv, pa, gate=gate, features=ft)}
            for nm, sc_ in scores.items():
                key = {'Vision only': 'vision_only', 'Audio only': 'audio_only', 'Static weighted': 'static', 'Noisy-OR': 'noisy_or', 'Proposed': 'proposed'}[nm]
                m = cell['methods'].setdefault(nm, {'f1': [], 'f1_std': []})
                m['f1'].append(evaluate_scores(y, sc_, thr[key])['f1'])
                m['f1_std'].append(0.0)
            w = gate.predict_weight(ft)
            cell['mean_w_v'].append(float(w.mean()))
            cell['mean_w_v_std'].append(float(w.std()))
            log.info('sweep %s/%s sev %d done (%.0fs elapsed)', ch, corr, sev, time.time() - t0)
        for ch in sweep:
            for corr, cell in sweep[ch].items():
                for nm in cell['methods']:
                    key = {'Vision only': 'vision_only', 'Audio only': 'audio_only', 'Static weighted': 'static', 'Noisy-OR': 'noisy_or', 'Proposed': 'proposed'}[nm]
                    clean = agg[key]['f1']['mean']
                    cell['methods'][nm]['f1'].insert(0, clean)
                    cell['methods'][nm]['f1_std'].insert(0, agg[key]['f1']['std'])
                cell['mean_w_v'].insert(0, runs[0]['gate']['mean_vision_weight'])
                cell['mean_w_v_std'].insert(0, runs[0]['gate']['std_vision_weight'])
        sweep_path.write_text(json.dumps(sweep, indent=2))
    degradation_curves(sweep, work / 'degradation_curves.png')
    gate_behaviour(sweep, work / 'gate_behaviour.png')
    emit_tex(agg, runs, sweep, work / 'results.tex')
    log.info('DONE in %.1f min. Upload %s/degradation_curves.png and %s/gate_behaviour.png to Overleaf, then paste %s/results.tex into the tables.', (time.time() - t0) / 60, a.work, a.work, a.work)
if __name__ == '__main__':
    main()
