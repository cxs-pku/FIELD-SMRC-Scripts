from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

PI = np.pi


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def tangent_basis(r: np.ndarray):
    er = unit(r)
    ref = np.array([0.0, 0.0, 1.0]) if abs(er[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(ref, er)
    if np.linalg.norm(t1) < 1e-12:
        ref = np.array([0.0, 1.0, 0.0])
        t1 = np.cross(ref, er)
    t1 = unit(t1)
    t2 = unit(np.cross(er, t1))
    return t1, t2, er


def fibonacci_hemisphere(n: int, radius: float) -> np.ndarray:
    pts = []
    golden = (1 + 5**0.5) / 2
    i = 0
    while len(pts) < n and i < 100000:
        z = 1 - (2 * i + 1) / (2 * n * 2)
        theta = 2 * PI * i / golden
        r = max(0.0, 1 - z * z) ** 0.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        if z > 0:
            pts.append(np.array([radius * x, radius * y, radius * z], dtype=float))
        i += 1
    return np.asarray(pts)


def sarvas_field_vector(rr: np.ndarray, q: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """Magnetic field vector at sensors for a current dipole in a spherical conductor.

    This is adapted from the Sarvas-style spherical-model MEG expression used in
    MNE-Python's forward computation internals. Positions are in meters, dipole
    moment is in A*m, and the return value is in Tesla.
    """
    this_poss = sensors
    a_vec = this_poss - rr[None, :]
    a = np.linalg.norm(a_vec, axis=1)
    r = np.linalg.norm(this_poss, axis=1)
    rr0 = this_poss @ rr
    ar = r * r - rr0
    ar0 = ar / np.maximum(a, 1e-15)
    F = a * (r * a + ar)
    gr = (a * a) / np.maximum(r, 1e-15) + ar0 + 2.0 * (a + r)
    g0 = a + 2 * r + ar0

    B = np.zeros((len(sensors), 3), dtype=float)
    rr_rep = np.repeat(rr[None, :], len(sensors), axis=0)
    v2 = np.cross(rr_rep, this_poss)
    for axis_idx, e in enumerate(np.eye(3)):
        re = this_poss @ e
        r0e = rr @ e
        g = (g0 * r0e - gr * re) / np.maximum(F * F, 1e-30)
        v1 = np.cross(rr_rep, np.repeat(e[None, :], len(sensors), axis=0))
        xx = (v1 / np.maximum(F[:, None], 1e-30) + v2 * g[:, None]) * 1e-7
        B[:, axis_idx] = (xx * q[None, :]).sum(axis=1)
    return B


def field_direction(polar_deg: float, azim_deg: float) -> np.ndarray:
    th = np.deg2rad(polar_deg)
    ph = np.deg2rad(azim_deg)
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], dtype=float)


def deadzone_laser(sensor_pos: np.ndarray, n_field: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    t1, t2, er = tangent_basis(sensor_pos)
    best = np.cross(er, n_field)
    if np.linalg.norm(best) < 1e-12:
        best = t1
    else:
        best = unit(best)
    worst = unit(np.cross(er, best))
    if mode == 'best':
        return best
    if mode == 'worst':
        return worst
    alpha = rng.uniform(0, 2 * PI)
    return unit(np.cos(alpha) * t1 + np.sin(alpha) * t2)


def local_field_artifacts(sensor_pos: np.ndarray, sensor_r: float, rng: np.random.Generator, grad_scale: float, azim_deg: float) -> float:
    x, y, z = sensor_pos / sensor_r
    phi = np.deg2rad(azim_deg)
    d1 = unit([np.cos(phi), np.sin(phi), 0.35])
    a1 = rng.normal(0, grad_scale * 0.8)
    a2 = rng.normal(0, grad_scale * 0.4)
    a3 = rng.normal(0, grad_scale * 0.25)
    return float(a1 * np.dot([x, y, z], d1) + a2 * (x * z) + a3 * (x * x - y * y))


def choose_subset_indices(count: int, n: int) -> np.ndarray:
    step = n / count
    idx = [int(k * step) for k in range(count)]
    return np.asarray(sorted(set(idx)))[:count]


@dataclass
class BenchmarkConfig:
    sensor_r: float = 0.10
    n_sensors: int = 64
    candidate_shells: tuple = ((0.06, 32), (0.07, 48), (0.08, 64))
    source_r_min: float = 0.06
    source_r_max: float = 0.08
    source_z_min: float = 0.20
    train_counts: tuple = (16, 32, 64)
    ood_counts: tuple = (16, 32, 64)
    intrinsic_levels: tuple = (5.0, 10.0, 20.0)
    dead_modes: tuple = ('best', 'random', 'worst')
    dipole_moment: float = 10e-9


class GeoMEGBenchmark:
    def __init__(self, cfg: BenchmarkConfig | None = None):
        self.cfg = cfg or BenchmarkConfig()
        self.sensors = fibonacci_hemisphere(self.cfg.n_sensors, self.cfg.sensor_r)
        cand = []
        for rad, n in self.cfg.candidate_shells:
            cand.extend(fibonacci_hemisphere(n, rad))
        self.candidates = np.asarray(cand)
        self.candidate_fields = self._precompute_candidate_fields()
        self.feature_names = None

    def _precompute_candidate_fields(self) -> np.ndarray:
        fields = np.zeros((len(self.candidates), 2, len(self.sensors), 3), dtype=np.float32)
        for j, r in enumerate(self.candidates):
            t1, t2, _ = tangent_basis(r)
            for b, qdir in enumerate([t1, t2]):
                fields[j, b] = sarvas_field_vector(r, qdir * self.cfg.dipole_moment, self.sensors) * 1e15
        return fields

    def sample_continuous_source(self, rng: np.random.Generator):
        rad = rng.uniform(self.cfg.source_r_min, self.cfg.source_r_max)
        z = rng.uniform(self.cfg.source_z_min, 1.0)
        phi = rng.uniform(0, 2 * PI)
        rxy = (1 - z * z) ** 0.5
        pos = np.array([rxy * np.cos(phi), rxy * np.sin(phi), z]) * rad
        t1, t2, _ = tangent_basis(pos)
        ang = rng.uniform(0, 2 * PI)
        qdir = np.cos(ang) * t1 + np.sin(ang) * t2
        return pos, qdir * self.cfg.dipole_moment

    def simulate_sample(self, rng: np.random.Generator, split: str = 'train'):
        pos, q = self.sample_continuous_source(rng)
        if split == 'ood':
            polar_deg = rng.uniform(75, 105)
            grad_scale = rng.uniform(5, 10)
            ext_white = rng.uniform(4, 9)
            count = int(rng.choice(self.cfg.ood_counts))
        else:
            polar_deg = rng.uniform(0, 70)
            grad_scale = rng.uniform(0, 6)
            ext_white = rng.uniform(0, 6)
            count = int(rng.choice(self.cfg.train_counts))
        azim_deg = rng.uniform(0, 360)
        n_field = field_direction(polar_deg, azim_deg)
        intrinsic = float(rng.choice(self.cfg.intrinsic_levels))
        dead_mode = str(rng.choice(self.cfg.dead_modes))
        active = choose_subset_indices(count, self.cfg.n_sensors)
        Bvec = sarvas_field_vector(pos, q, self.sensors) * 1e15
        signal = Bvec @ n_field
        y = np.zeros(self.cfg.n_sensors, dtype=np.float32)
        mask = np.zeros(self.cfg.n_sensors, dtype=np.float32)
        sin2_vals = []
        for i in active:
            mask[i] = 1
            laser = deadzone_laser(self.sensors[i], n_field, dead_mode, rng)
            s = float(np.clip(1 - float(np.dot(laser, n_field) ** 2), 0.05, 1.0))
            sin2_vals.append(s)
            sigma = ((intrinsic**2) / s + ext_white**2) ** 0.5
            gain = 1 - 0.18 * (1 - s) + rng.normal(0, 0.02)
            artifact = local_field_artifacts(self.sensors[i], self.cfg.sensor_r, rng, grad_scale, azim_deg)
            y[i] = gain * signal[i] + artifact + rng.normal(0, sigma)
        meta = {
            'count': count,
            'intrinsic': intrinsic,
            'ext_white': float(ext_white),
            'grad_scale': float(grad_scale),
            'polar_deg': float(polar_deg),
            'azim_deg': float(azim_deg),
            'dead_mode': dead_mode,
            'sin2mean': float(np.mean(sin2_vals)),
            'nField': n_field,
            'source_r': float(np.linalg.norm(pos)),
        }
        return pos.astype(np.float32), y, mask, meta

    def feature_pack(self, y: np.ndarray, mask: np.ndarray, meta: dict):
        n_field = meta['nField']
        proj = np.tensordot(self.candidate_fields, n_field, axes=([3], [0]))  # G,2,S
        idx = np.where(mask > 0.5)[0]
        H = proj[:, :, idx]
        c1 = H[:, 0, :]
        c2 = H[:, 1, :]
        yy = y[idx]
        lam = 1e-3
        a11 = (c1 * c1).sum(axis=1) + lam
        a22 = (c2 * c2).sum(axis=1) + lam
        a12 = (c1 * c2).sum(axis=1)
        b1 = (c1 * yy).sum(axis=1)
        b2 = (c2 * yy).sum(axis=1)
        det = a11 * a22 - a12 * a12 + 1e-12
        x1 = (a22 * b1 - a12 * b2) / det
        x2 = (a11 * b2 - a12 * b1) / det
        pred = c1 * x1[:, None] + c2 * x2[:, None]
        rss = ((yy[None, :] - pred) ** 2).sum(axis=1)
        fitE = x1 * x1 + x2 * x2
        corr = (pred * yy[None, :]).sum(axis=1)
        score = -np.log(rss + 1e-6)
        fit = np.log(fitE + 1e-12)
        co = np.sign(corr) * np.log(np.abs(corr) + 1e-9)
        top = np.argsort(rss)[:10]
        w = np.exp(-(rss[top] - rss[top].min()) / (np.std(rss[top]) + 1e-6))
        w = w / w.sum()
        cent = (self.candidates[top] * w[:, None]).sum(axis=0)

        top_pack = []
        top_names = []
        for rank, j in enumerate(top, 1):
            top_pack.extend([score[j], fit[j], co[j], self.candidates[j, 0], self.candidates[j, 1], self.candidates[j, 2]])
            top_names.extend([
                f'top{rank}_score', f'top{rank}_fit', f'top{rank}_corr', f'top{rank}_x', f'top{rank}_y', f'top{rank}_z'
            ])

        count_vec = [1 if meta['count'] == v else 0 for v in self.cfg.train_counts]
        dead_vec = [1 if meta['dead_mode'] == v else 0 for v in self.cfg.dead_modes]
        nvec = np.array([
            meta['count'] / max(self.cfg.train_counts),
            meta['intrinsic'] / max(self.cfg.intrinsic_levels),
            meta['ext_white'] / 9.0,
            meta['grad_scale'] / 10.0,
            meta['polar_deg'] / 105.0,
            meta['azim_deg'] / 360.0,
            meta['sin2mean'],
            *count_vec,
            *dead_vec,
            *meta['nField'],
        ], dtype=np.float32)
        nuisance_names = [
            'count_norm', 'intrinsic_norm', 'ext_noise_norm', 'grad_norm', 'polar_norm', 'azim_norm', 'sin2_mean',
            'count16', 'count32', 'count64', 'dead_best', 'dead_random', 'dead_worst', 'ng_x', 'ng_y', 'ng_z'
        ]
        full = np.concatenate([np.asarray(top_pack, dtype=np.float32), cent.astype(np.float32), nvec])
        no_cent = np.concatenate([np.asarray(top_pack, dtype=np.float32), nvec])
        cent_only = np.concatenate([cent.astype(np.float32), nvec])
        raw = np.concatenate([y.astype(np.float32), mask.astype(np.float32), nvec])
        centroid = cent.astype(np.float32)
        grid = self.candidates[np.argmin(rss)].astype(np.float32)

        m = len(idx)
        G = len(self.candidates)
        L = np.zeros((m, 2 * G), dtype=np.float32)
        for j in range(G):
            L[:, 2 * j] = H[j, 0, :]
            L[:, 2 * j + 1] = H[j, 1, :]
        A = L @ L.T + np.eye(m, dtype=np.float32) * 1e-2
        beta = L.T @ np.linalg.solve(A, yy.astype(np.float32))
        power = np.sqrt(beta[0::2] ** 2 + beta[1::2] ** 2)
        mne = self.candidates[np.argmax(power)].astype(np.float32)

        score_map = score.astype(np.float32)
        if self.feature_names is None:
            self.feature_names = top_names + ['cent_x', 'cent_y', 'cent_z'] + nuisance_names
        return {
            'full': full,
            'no_cent': no_cent,
            'cent_only': cent_only,
            'raw': raw,
            'centroid': centroid,
            'grid': grid,
            'mne': mne,
            'score_map': score_map,
            'top_idx': top.astype(int),
        }

    def build_dataset(self, n: int, split: str, seed: int):
        rng = np.random.default_rng(seed)
        full, no_cent, cent_only, raw = [], [], [], []
        truth, centroids, grids, mnes, metas, y_list, mask_list, score_maps, top_indices = [], [], [], [], [], [], [], [], []
        for _ in range(n):
            pos, y, mask, meta = self.simulate_sample(rng, split)
            feats = self.feature_pack(y, mask, meta)
            full.append(feats['full'])
            no_cent.append(feats['no_cent'])
            cent_only.append(feats['cent_only'])
            raw.append(feats['raw'])
            truth.append(pos)
            centroids.append(feats['centroid'])
            grids.append(feats['grid'])
            mnes.append(feats['mne'])
            score_maps.append(feats['score_map'])
            top_indices.append(feats['top_idx'])
            y_list.append(y)
            mask_list.append(mask)
            meta_out = dict(meta)
            meta_out['nField_x'] = float(meta['nField'][0])
            meta_out['nField_y'] = float(meta['nField'][1])
            meta_out['nField_z'] = float(meta['nField'][2])
            del meta_out['nField']
            metas.append(meta_out)
        return {
            'full': np.stack(full),
            'no_cent': np.stack(no_cent),
            'cent_only': np.stack(cent_only),
            'raw': np.stack(raw),
            'truth': np.stack(truth),
            'centroid': np.stack(centroids),
            'grid': np.stack(grids),
            'mne': np.stack(mnes),
            'meta': metas,
            'y': np.stack(y_list),
            'mask': np.stack(mask_list),
            'score_map': np.stack(score_maps),
            'top_idx': np.stack(top_indices),
        }


def regression_metrics(err_mm: np.ndarray) -> dict:
    return {
        'mean_mm': float(np.mean(err_mm)),
        'median_mm': float(np.median(err_mm)),
        'sd_mm': float(np.std(err_mm)),
        'acc_10mm': float(np.mean(err_mm <= 10)),
        'acc_15mm': float(np.mean(err_mm <= 15)),
        'fail_gt25mm': float(np.mean(err_mm > 25)),
    }


def fit_regressor(X: np.ndarray, Y: np.ndarray, *, n_estimators: int = 120, min_samples_leaf: int = 2, random_state: int = 0):
    reg = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=1,
    )
    reg.fit(X, Y)
    return reg


def compute_errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - truth, axis=1) * 1000.0


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
