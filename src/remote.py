"""Field-scale rice stress from MODIS NDVI time series, plus weather correlation (pure NumPy).

What this can and cannot show: a 250 m MODIS pixel (~5.4 ha) covers thousands of plants, so
leaf lesions are invisible. We detect *canopy stress anomalies*: rice pixels whose NDVI is
unusually low for that time of year compared with the same 16-day slot in other years. Disease
is one cause; drought, flood, late planting and harvest timing are others. Outputs are therefore
"stress that may indicate disease", never a disease diagnosis.

Chain (mirrors the leaf pipeline): QA mask -> rice mask (Otsu on seasonal amplitude + connected
components) -> leave-one-year-out robust z-score per pixel and 16-day slot -> stress mask
(threshold + component area filter) -> stressed-area fraction -> weather anomalies, lagged rank
correlation with a circular-shift permutation test, and a small k-NN.

All parameters in FieldConfig are engineering choices fixed before looking at the results.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import combinations
import warnings

import numpy as np

from .algorithms import components, otsu

NDVI_FILL = -3000
NDVI_SCALE = 1e-4
SLOTS_PER_YEAR = 23


@dataclass(frozen=True)
class FieldConfig:
    good_reliability: tuple = (0, 1)   # MOD13Q1 pixel_reliability: 0 good, 1 marginal, 2 snow, 3 cloudy
    min_valid_frac: float = .3         # a pixel needs >=30 % clear composites to be considered for the rice mask
    min_ref: int = 3                   # at least 3 other years with a clear value in the same slot
    sigma_floor: float = .03           # NDVI units; avoids huge z where the reference is nearly constant
    z_threshold: float = -1.5          # stressed if NDVI is >=1.5 robust SD below the other years' median
    min_area_px: int = 4               # 4 pixels ~ 21 ha; smaller blobs are treated as noise
    rice_min_area_px: int = 20         # rice-mask components smaller than ~1 km2 are dropped
    min_coverage: float = .3           # composite counted only if >=30 % of rice pixels are clear
    lags: tuple = (0, 1, 2, 3)         # weather window lags in composites (0-48 days before)
    knn_k: int = 7
    knn_lags: tuple = (1, 2)           # k-NN uses only weather that precedes the composite
    high_quantile: float = .75         # "high stress" = above the training years' 75th percentile
    outbreak_years: tuple = (2006, 2007)  # fixed from the literature before running (see docs)
    weather_vars: tuple = field(default=('RH2M', 'T2M', 'PRECTOTCORR', 'WET_DAYS'))


# ---------------------------------------------------------------- MODIS decoding

def decode_ndvi(raw, reliability, cfg=FieldConfig()):
    """int16 MOD13Q1 NDVI -> float NDVI with NaN for fill and non-clear pixels."""
    ndvi = raw.astype(float)*NDVI_SCALE
    bad = (raw == NDVI_FILL) | ~np.isin(reliability, cfg.good_reliability)
    ndvi[bad] = np.nan
    return ndvi


def slot_of(dates):
    """16-day composite slot 0..22 within the year, and the year, from ISO date strings."""
    d = [date.fromisoformat(str(x)) for x in dates]
    return (np.array([(x.timetuple().tm_yday-1)//16 for x in d]),
            np.array([x.year for x in d]))


# ---------------------------------------------------------------- rice mask

def seasonal_amplitude(ndvi):
    """Per-pixel p90-p10 NDVI over all clear composites, and the clear fraction."""
    valid = np.isfinite(ndvi).mean(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        amp = np.nanpercentile(ndvi, 90, 0)-np.nanpercentile(ndvi, 10, 0)
    return np.where(valid > 0, amp, np.nan), valid


def rice_mask(ndvi, cfg=FieldConfig()):
    """Paddy rice cycles flooded (NDVI<0.3) -> green (>0.6) two or three times a year, so its
    seasonal amplitude is large; orchards, towns and the river are low-amplitude. Otsu splits
    amplitude; connected components drop isolated pixels."""
    amp, valid = seasonal_amplitude(ndvi)
    ok = (valid >= cfg.min_valid_frac) & np.isfinite(amp)
    thr = otsu(amp[ok])
    mask = ok & (amp >= thr)
    labels, areas = components(mask)
    keep = np.flatnonzero(areas >= cfg.rice_min_area_px)
    keep = keep[keep > 0]
    return np.isin(labels, keep), {'amplitude_threshold': float(thr),
                                   'amplitude': amp, 'valid_frac': valid}


# ---------------------------------------------------------------- anomaly and stress

def loyo_zscore(ndvi, slots, years, cfg=FieldConfig()):
    """Robust z of each composite against the same slot in all *other* years (leave-one-year-out),
    so no composite is compared with itself. sigma = max(1.4826*MAD, sigma_floor)."""
    z = np.full(ndvi.shape, np.nan)
    for s in np.unique(slots):
        idx = np.flatnonzero(slots == s)
        for t in idx:
            ref = ndvi[idx[years[idx] != years[t]]]
            if len(ref) < cfg.min_ref:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n = np.isfinite(ref).sum(0)
                med = np.nanmedian(ref, 0)
                mad = np.nanmedian(np.abs(ref-med), 0)*1.4826
            sigma = np.maximum(mad, cfg.sigma_floor)
            zt = (ndvi[t]-med)/sigma
            zt[n < cfg.min_ref] = np.nan
            z[t] = zt
    return z


def stress_mask(z, rice, cfg=FieldConfig()):
    """Threshold z inside the rice mask, keep 4-connected blobs of >= min_area_px pixels."""
    raw = rice & np.isfinite(z) & (z <= cfg.z_threshold)
    labels, areas = components(raw)
    keep = np.flatnonzero(areas >= cfg.min_area_px)
    return np.isin(labels, keep[keep > 0])


def stress_series(z, rice, cfg=FieldConfig()):
    """Per composite: stressed fraction of *clear* rice pixels, clear coverage, and the masks.
    Composites below min_coverage get NaN fraction but are still counted in the report."""
    n_rice = rice.sum()
    frac, cover, masks = [], [], []
    for zt in z:
        clear = rice & np.isfinite(zt)
        m = stress_mask(zt, rice, cfg)
        c = clear.sum()/n_rice if n_rice else 0.
        cover.append(c)
        frac.append(m.sum()/clear.sum() if c >= cfg.min_coverage and clear.sum() else np.nan)
        masks.append(m)
    return np.array(frac), np.array(cover), np.array(masks)


def severity_label(fraction, thresholds=(.01, .05, .20),
                   labels=('none', 'light', 'moderate', 'severe')):
    """Same area bins as the leaf pipeline (data/severity_rules.json), applied to field area."""
    if not np.isfinite(fraction):
        return 'no_data'
    return labels[int(np.searchsorted(thresholds, fraction, side='right'))]


# ---------------------------------------------------------------- weather

def power_daily(power_json):
    """NASA POWER daily JSON -> (dates, {param: array}) with fill values as NaN."""
    params = power_json['properties']['parameter']
    fill = power_json['header'].get('fill_value', -999.)
    keys = sorted(next(iter(params.values())).keys())
    out = {}
    for name, series in params.items():
        v = np.array([series[k] for k in keys], float)
        v[v == fill] = np.nan
        out[name] = v
    return np.array([date(int(k[:4]), int(k[4:6]), int(k[6:])) for k in keys]), out


def weather_windows(composite_dates, wdates, wvals, days=16):
    """Aggregate daily weather over each composite's 16-day window [start, start+16):
    mean RH2M/T2M, total PRECTOTCORR, and WET_DAYS = days with >=1 mm (standard wet-day cut)."""
    index = {d: i for i, d in enumerate(wdates)}
    out = {k: [] for k in ('RH2M', 'T2M', 'PRECTOTCORR', 'WET_DAYS')}
    for s in composite_dates:
        s = date.fromisoformat(str(s))
        idx = [index[s+timedelta(i)] for i in range(days) if s+timedelta(i) in index]
        enough = len(idx) >= days//2
        sel = lambda k: wvals[k][idx]
        out['RH2M'].append(np.nanmean(sel('RH2M')) if enough else np.nan)
        out['T2M'].append(np.nanmean(sel('T2M')) if enough else np.nan)
        out['PRECTOTCORR'].append(np.nansum(sel('PRECTOTCORR')) if enough else np.nan)
        out['WET_DAYS'].append(float((sel('PRECTOTCORR') >= 1).sum()) if enough else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def loyo_anomaly_1d(x, slots, years):
    """Seasonal anomaly of a scalar series: (x - mean of same slot in other years) / their SD.
    Removes the seasonal cycle so correlations are not just 'rainy season vs dry season'."""
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        ref = x[(slots == slots[t]) & (years != years[t]) & np.isfinite(x)]
        if len(ref) >= 3 and np.isfinite(x[t]):
            out[t] = (x[t]-ref.mean())/max(ref.std(ddof=1), 1e-9)
    return out


# ---------------------------------------------------------------- statistics

def rank(x):
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x)+1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            r[tie] = r[tie].mean()
    return r


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    ra, rb = rank(a[ok]), rank(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def lag(x, k):
    """x shifted so out[t] = x[t-k] (NaN-padded)."""
    out = np.full(len(x), np.nan)
    out[k:] = x[:len(x)-k] if k else x
    return out


def circular_shift_test(y, x, min_shift=SLOTS_PER_YEAR):
    """Two-sided p for Spearman(y, x) against all circular shifts of x by >= one year.
    Shifting keeps each series' own autocorrelation, unlike shuffling single points."""
    r0, n = spearman(y, x)
    if not np.isfinite(r0):
        return dict(r=None, n=n, p=None)
    null = [spearman(y, np.roll(x, k))[0] for k in range(min_shift, len(x)-min_shift+1)]
    null = np.array([v for v in null if np.isfinite(v)])
    p = (np.sum(np.abs(null) >= abs(r0))+1)/(len(null)+1)
    return dict(r=r0, n=n, p=float(p), n_shifts=int(len(null)))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (None passes through)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    adj, running = [None]*len(pvalues), 0.
    for rank_i, i in enumerate(order):
        running = max(running, min(1., (len(order)-rank_i)*pvalues[i]))
        adj[i] = running
    return adj


def year_permutation_test(annual, target_years):
    """Exact test: is the mean over target_years higher than for every other set of the same size?"""
    years = sorted(y for y, v in annual.items() if np.isfinite(v))
    obs = np.mean([annual[y] for y in target_years])
    sets = list(combinations(years, len(target_years)))
    hits = sum(np.mean([annual[y] for y in s]) >= obs-1e-12 for s in sets)
    return dict(observed=float(obs), others_mean=float(np.mean([annual[y] for y in years
                                                                  if y not in target_years])),
                n_sets=len(sets), p_one_sided=hits/len(sets),
                rank_of_target=int(1+sum(np.mean([annual[y] for y in s]) > obs+1e-12 for s in sets)))


# ---------------------------------------------------------------- k-NN (train-only scaling)

def knn_predict(x_train, y_train, x_test, k):
    mu, sd = x_train.mean(0), x_train.std(0)
    sd[sd == 0] = 1
    a, b = (x_train-mu)/sd, (x_test-mu)/sd
    d = ((b[:, None, :]-a[None])**2).sum(-1)
    near = np.argsort(d, 1, kind='stable')[:, :k]
    return (y_train[near].mean(1) > .5).astype(int)


def balanced_accuracy(y, p):
    rec = [np.mean(p[y == c] == c) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(rec)) if rec else None


def knn_leave_one_year_out(frac, weather_anom, years, cfg=FieldConfig()):
    """Predict 'high stress' at composite t from weather anomalies at lags cfg.knn_lags.
    Folds are whole years; the high-stress cut is the training years' quantile only.
    Baselines: always-low (majority) and persistence (yesterday's satellite label)."""
    cols = [lag(weather_anom[v], k) for v in cfg.weather_vars for k in cfg.knn_lags]
    x = np.column_stack(cols)
    prev = lag(frac, 1)
    ok = np.isfinite(frac) & np.isfinite(x).all(1)
    y_true, y_knn, y_major, y_persist, y_year = [], [], [], [], []
    for yr in np.unique(years[ok]):
        tr, te = ok & (years != yr), ok & (years == yr)
        if tr.sum() < cfg.knn_k or not te.any():
            continue
        cut = np.quantile(frac[tr], cfg.high_quantile)
        ytr, yte = (frac[tr] > cut).astype(int), (frac[te] > cut).astype(int)
        y_true += list(yte)
        y_knn += list(knn_predict(x[tr], ytr, x[te], cfg.knn_k))
        y_major += [0]*len(yte)
        y_persist += list(np.where(np.isfinite(prev[te]), prev[te] > cut, 0).astype(int))
        y_year += [int(yr)]*len(yte)
    y_true, y_knn, y_major, y_persist = map(np.array, (y_true, y_knn, y_major, y_persist))
    return dict(
        n=int(len(y_true)), n_high=int(y_true.sum()), features=[f'{v}_lag{k}' for v in cfg.weather_vars
                                                               for k in cfg.knn_lags],
        k=cfg.knn_k, balanced_accuracy={'knn_weather': balanced_accuracy(y_true, y_knn),
                                        'majority': balanced_accuracy(y_true, y_major),
                                        'persistence_satellite': balanced_accuracy(y_true, y_persist)},
        per_year=[dict(year=int(y), n=int((np.array(y_year) == y).sum()),
                       n_high=int(y_true[np.array(y_year) == y].sum()),
                       knn_hits=int(((y_knn == 1) & (y_true == 1))[np.array(y_year) == y].sum()))
                  for y in sorted(set(y_year))])
