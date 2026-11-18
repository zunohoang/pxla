"""Field-scale MODIS stress module (src/remote.py) on synthetic series with known answers."""
from datetime import date, timedelta

import numpy as np
from scipy import stats

from src.remote import (FieldConfig, circular_shift_test, decode_ndvi, holm, knn_leave_one_year_out,
                        loyo_zscore, rank, rice_mask, severity_label, slot_of, spearman,
                        stress_series, weather_windows, year_permutation_test)


def synthetic_field(years=range(2000, 2008), h=30, w=30, seed=0):
    """Rice block (left half) with a 2-crop NDVI cycle; orchard (right half) flat at 0.7.
    In 2005 a 6x6 rice patch drops 0.25 NDVI in slots 5-8: a known injected stress."""
    rng = np.random.default_rng(seed)
    dates = [date(y, 1, 1)+timedelta(16*s) for y in years for s in range(23)]
    slots, yrs = slot_of([d.isoformat() for d in dates])
    t = slots/23
    rice_curve = .45-.3*np.cos(4*np.pi*t)  # 0.15 .. 0.75 twice a year
    ndvi = np.empty((len(dates), h, w))
    ndvi[:, :, :w//2] = rice_curve[:, None, None]
    ndvi[:, :, w//2:] = .7
    ndvi += rng.normal(0, .02, ndvi.shape)
    hit = (yrs == 2005) & (slots >= 5) & (slots <= 8)
    ndvi[np.ix_(hit, range(10, 16), range(4, 10))] -= .25
    truth = np.zeros((h, w), bool)
    truth[10:16, 4:10] = True
    return dates, slots, yrs, ndvi, truth, hit


def test_decode_masks_fill_and_cloud():
    raw = np.array([[5000, -3000, 7000]], np.int16)
    rel = np.array([[0, 0, 3]], np.int8)
    out = decode_ndvi(raw, rel)
    assert out[0, 0] == .5 and np.isnan(out[0, 1]) and np.isnan(out[0, 2])


def test_rice_mask_separates_cycling_from_flat_vegetation():
    _, _, _, ndvi, _, _ = synthetic_field()
    mask, info = rice_mask(ndvi)
    assert mask[:, :15].mean() > .95 and mask[:, 15:].mean() < .05
    assert 0 < info['amplitude_threshold'] < .6


def test_injected_stress_is_recovered_and_only_there():
    _, slots, yrs, ndvi, truth, hit = synthetic_field()
    rice, _ = rice_mask(ndvi)
    z = loyo_zscore(ndvi, slots, yrs)
    frac, cover, masks = stress_series(z, rice)
    found = masks[hit].any(0)
    iou = (found & truth).sum()/(found | truth).sum()
    assert iou > .9
    assert np.nanmax(frac[~hit]) < .01 and np.nanmin(frac[hit]) > .05
    assert np.allclose(cover, 1)


def test_low_coverage_composite_is_nan_but_kept():
    _, slots, yrs, ndvi, _, _ = synthetic_field()
    rice, _ = rice_mask(ndvi)
    ndvi[3, :, :12] = np.nan  # cloud over 80 % of the rice
    frac, cover, _ = stress_series(loyo_zscore(ndvi, slots, yrs), rice)
    assert len(frac) == len(ndvi) and np.isnan(frac[3]) and cover[3] < .3


def test_rank_and_spearman_match_scipy():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 6, 40).astype(float)
    b = a+rng.normal(0, 2, 40)
    assert np.allclose(rank(a), stats.rankdata(a))
    assert abs(spearman(a, b)[0]-stats.spearmanr(a, b).statistic) < 1e-12


def test_circular_shift_test_detects_lagged_link_and_not_noise():
    rng = np.random.default_rng(4)
    x = rng.normal(size=230)
    y = np.r_[np.zeros(2), x[:-2]]+rng.normal(0, .5, 230)
    from src.remote import lag
    assert circular_shift_test(y, lag(x, 2))['p'] < .01
    assert circular_shift_test(y, rng.normal(size=230))['p'] > .01


def test_holm_and_year_permutation():
    assert holm([.01, .04, None, .03]) == [.03, .06, None, .06]
    annual = {2000+i: v for i, v in enumerate([.01, .02, .01, .02, .01, .015, .09, .08])}
    r = year_permutation_test(annual, (2006, 2007))
    assert r['n_sets'] == 28 and r['rank_of_target'] == 1 and abs(r['p_one_sided']-1/28) < 1e-12


def test_weather_windows_and_severity():
    days = [date(2006, 1, 1)+timedelta(i) for i in range(40)]
    vals = {'RH2M': np.full(40, 80.), 'T2M': np.full(40, 27.), 'PRECTOTCORR': np.r_[np.full(5, 3.), np.zeros(35)]}
    w = weather_windows(['2006-01-01', '2006-01-17'], np.array(days), vals)
    assert w['PRECTOTCORR'][0] == 15 and w['WET_DAYS'][0] == 5 and w['WET_DAYS'][1] == 0
    assert [severity_label(v) for v in (0, .02, .1, .5, np.nan)] == ['none', 'light', 'moderate', 'severe', 'no_data']


def test_knn_beats_majority_when_weather_predicts_stress():
    rng = np.random.default_rng(5)
    n = 23*8
    years = np.repeat(np.arange(2000, 2008), 23)
    anom = {v: rng.normal(size=n) for v in FieldConfig().weather_vars}
    frac = np.r_[0, anom['RH2M'][:-1]]*.02+.05+rng.normal(0, .002, n)  # stress follows RH at lag 1
    r = knn_leave_one_year_out(frac, anom, years)
    assert r['balanced_accuracy']['majority'] == .5
    assert r['balanced_accuracy']['knn_weather'] > .7


def test_run_field_end_to_end_on_synthetic_site(tmp_path):
    import json
    from run_field import run
    dates, slots, yrs, ndvi, _, _ = synthetic_field(years=range(2000, 2009), h=24, w=24)
    site = tmp_path/'site'
    site.mkdir()
    raw = np.round(np.clip(ndvi, -.2, 1)/1e-4).astype(np.int16)
    np.savez_compressed(site/'modis.npz', dates=np.array([d.isoformat() for d in dates]), ndvi_raw=raw,
                        reliability=np.zeros(raw.shape, np.int8), cellsize=231.656, xllcorner=0., yllcorner=0.)
    days = [date(2000, 1, 1)+timedelta(i) for i in range(9*366)]
    rng = np.random.default_rng(0)
    param = {k: {d.strftime('%Y%m%d'): float(v) for d, v in zip(days, rng.normal(m, s, len(days)))}
             for k, m, s in (('RH2M', 80, 5), ('T2M', 27, 1), ('PRECTOTCORR', 5, 3))}
    (site/'power.json').write_text(json.dumps({'properties': {'parameter': param}, 'header': {'fill_value': -999}}))
    r = run(site, tmp_path/'out')
    assert r['n_composites'] == len(dates) and r['composites_usable'] > 0
    assert len(r['correlations']) == 16 and all('p_holm' in t for t in r['correlations'])
    assert (tmp_path/'out'/'field_analysis.json').exists() and (tmp_path/'out'/'field_stress_maps.png').exists()
