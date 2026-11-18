"""Field-scale rice stress (MODIS MOD13Q1) vs NASA POWER weather, An Giang / Dong Thap 2000-2010.

Input: data/raw/modis/<site>/{modis.npz, power.json} from scripts/download_modis_rice.py.
Nothing is tuned: FieldConfig defaults are fixed in src/remote.py. The outbreak test uses years
fixed from the literature (2006, 2007) before running. Cloudy composites stay in the counts.

There is no pixel-level disease ground truth. Results are stress anomalies that *may* indicate
disease; confounders (flood, drought, cropping-calendar shifts, harvest timing) are not removed.

Outputs: results/field/field_analysis.json and results/field/*.png
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_experiments import save_json
from src.remote import (FieldConfig, circular_shift_test, decode_ndvi, holm, knn_leave_one_year_out,
                        lag, loyo_anomaly_1d, loyo_zscore, power_daily, rice_mask, severity_label,
                        slot_of, stress_series, weather_windows, year_permutation_test)

VAR_VI = {'RH2M': 'Độ ẩm TB (%)', 'T2M': 'Nhiệt độ TB (°C)', 'PRECTOTCORR': 'Tổng mưa (mm)',
          'WET_DAYS': 'Số ngày mưa ≥1 mm'}


def nanmean_or_none(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def run(site_dir, out, cfg=FieldConfig()):
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(site_dir/'modis.npz')
    dates = [str(x) for x in d['dates']]
    ndvi = decode_ndvi(d['ndvi_raw'], d['reliability'], cfg)
    slots, years = slot_of(dates)
    pixel_ha = float(d['cellsize'])**2/1e4

    rice, info = rice_mask(ndvi, cfg)
    z = loyo_zscore(ndvi, slots, years, cfg)
    frac, cover, masks = stress_series(z, rice, cfg)

    wdates, wvals = power_daily(json.loads((site_dir/'power.json').read_text(encoding='utf-8')))
    weather = weather_windows(dates, wdates, wvals)
    anom = {k: loyo_anomaly_1d(v, slots, years) for k, v in weather.items()}

    # lagged correlation stress(t) vs weather anomaly(t-lag), circular-shift permutation p, Holm
    tests = []
    for v in cfg.weather_vars:
        for k in cfg.lags:
            r = circular_shift_test(frac, lag(anom[v], k))
            tests.append(dict(variable=v, lag_composites=k, lag_days=16*k, **r))
    for t, p in zip(tests, holm([t['p'] for t in tests])):
        t['p_holm'] = p

    annual = {int(y): nanmean_or_none(frac[years == y]) for y in np.unique(years)}
    annual_ok = {y: v for y, v in annual.items() if v is not None}
    outbreak = year_permutation_test(annual_ok, cfg.outbreak_years)
    knn = knn_leave_one_year_out(frac, anom, years, cfg)

    per_year = []
    for y in np.unique(years):
        sel = years == y
        f = frac[sel]
        per_year.append(dict(
            year=int(y), composites=int(sel.sum()), usable=int(np.isfinite(f).sum()),
            mean_stressed_fraction=nanmean_or_none(f), max_stressed_fraction=float(np.nanmax(f))
            if np.isfinite(f).any() else None,
            mean_clear_coverage=float(cover[sel].mean()),
            severity_counts={s: int(sum(severity_label(x) == s for x in f))
                             for s in ('none', 'light', 'moderate', 'severe', 'no_data')}))

    result = dict(
        site=site_dir.name, config=asdict(cfg), period=[dates[0], dates[-1]],
        n_composites=len(dates), grid=list(rice.shape), pixel_area_ha=pixel_ha,
        rice_mask=dict(pixels=int(rice.sum()), area_km2=float(rice.sum()*pixel_ha/100),
                       fraction_of_box=float(rice.mean()),
                       amplitude_threshold=info['amplitude_threshold']),
        composites_usable=int(np.isfinite(frac).sum()),
        composites_dropped_low_coverage=int(np.isnan(frac).sum()),
        per_year=per_year,
        outbreak_test=dict(target_years=list(cfg.outbreak_years), **outbreak,
                           note='Exact permutation over all same-size sets of years with data; '
                                'annual value = mean stressed fraction of usable composites.'),
        correlations=tests,
        knn=knn,
        caveats=[
            'No pixel-level disease ground truth exists; stress != disease.',
            'POWER weather is one ~55 x 68 km cell: the same value for every pixel, so only '
            'temporal correlation is meaningful.',
            'Leave-one-year-out baselines include the outbreak years in other years\' references; '
            'the robust median limits but does not remove this.',
            'Cropping calendar changed in An Giang during the 2000s (third crop behind dykes); '
            'slot anomalies can reflect calendar shifts.',
            'The 2000 and 2001 Mekong floods fall in the record and can appear as stress.',
        ])
    save_json(out/'field_analysis.json', result)
    figures(out, dates, years, ndvi, rice, info, z, frac, cover, masks, annual, tests, cfg)
    return result


def figures(out, dates, years, ndvi, rice, info, z, frac, cover, masks, annual, tests, cfg):
    t = np.array(dates, dtype='datetime64[D]')
    # 1. rice mask
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.nanmedian(ndvi, 0), cmap='YlGn', vmin=0, vmax=.9)
    ax[0].set_title('NDVI trung vị 2000-2010')
    im = ax[1].imshow(info['amplitude'], cmap='magma')
    ax[1].set_title(f'Biên độ mùa vụ (p90-p10), Otsu={info["amplitude_threshold"]:.2f}')
    plt.colorbar(im, ax=ax[1], fraction=.046)
    ax[2].imshow(rice, cmap='Greens')
    ax[2].set_title(f'Mặt nạ lúa ({rice.mean()*100:.0f}% khung)')
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(out/'field_rice_mask.png', dpi=150)
    plt.close(fig)

    # 2. time series
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].axvspan(np.datetime64('2005-01-01'), np.datetime64('2009-01-01'), color='#f4d9c6', label='Dịch rầy nâu 2005-2008 (tài liệu)')
    ax[0].axvspan(np.datetime64(f'{cfg.outbreak_years[0]}-01-01'), np.datetime64(f'{cfg.outbreak_years[-1]+1}-01-01'), color='#e8a77f', label='Năm kiểm định 2006-2007')
    ax[0].plot(t, frac*100, '.-', color='#b03a2e', lw=1, ms=3)
    ax[0].set_ylabel('% diện tích lúa bất thường')
    ax[0].legend(loc='upper left', fontsize=8)
    ax[1].bar(t, cover*100, width=np.timedelta64(12, 'D'), color='#7f8c8d')
    ax[1].axhline(cfg.min_coverage*100, color='k', ls='--', lw=.8)
    ax[1].set_ylabel('% pixel lúa không mây')
    fig.tight_layout()
    fig.savefig(out/'field_stress_timeseries.png', dpi=150)
    plt.close(fig)

    # 3. annual bars
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ys = sorted(annual)
    vals = [annual[y]*100 if annual[y] is not None else 0 for y in ys]
    ax.bar([str(y) for y in ys], vals, color=['#c0392b' if y in cfg.outbreak_years else '#95a5a6' for y in ys])
    ax.set_ylabel('% bất thường TB năm')
    fig.tight_layout()
    fig.savefig(out/'field_stress_annual.png', dpi=150)
    plt.close(fig)

    # 4. stress frequency maps: share of usable composites in which each rice pixel was stressed
    show = [y for y in (2003, 2004, 2006, 2007) if y in set(years)]
    fig, ax = plt.subplots(1, len(show), figsize=(3.2*len(show), 3.4))
    for a, y in zip(np.atleast_1d(ax), show):
        sel = (years == y) & np.isfinite(frac)
        freq = masks[sel].mean(0) if sel.any() else np.zeros(rice.shape)
        a.imshow(np.where(rice, freq, np.nan), cmap='Reds', vmin=0, vmax=.3)
        a.set_title(f'{y}: tần suất bất thường')
        a.axis('off')
    fig.tight_layout()
    fig.savefig(out/'field_stress_maps.png', dpi=150)
    plt.close(fig)

    # 5. lag correlations
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, v in enumerate(cfg.weather_vars):
        rows = [r for r in tests if r['variable'] == v]
        xs = [r['lag_days'] + (i-1.5)*2.5 for r in rows]
        rs = [r['r'] if r['r'] is not None else np.nan for r in rows]
        ax.bar(xs, rs, width=2.4, label=VAR_VI[v])
        for x, r in zip(xs, rows):
            if r['p_holm'] is not None and r['p_holm'] < .05:
                ax.text(x, r['r'], '*', ha='center')
    ax.axhline(0, color='k', lw=.6)
    ax.set_xlabel('Độ trễ thời tiết (ngày)')
    ax.set_ylabel('Spearman r')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'field_weather_lags.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site-dir', type=Path, default=Path('data/raw/modis/angiang'))
    p.add_argument('--out', type=Path, default=Path('results/field'))
    a = p.parse_args()
    r = run(a.site_dir, a.out)
    print(json.dumps({k: r[k] for k in ('rice_mask', 'composites_usable', 'composites_dropped_low_coverage',
                                        'outbreak_test')}, ensure_ascii=False, indent=1))
    for t in r['correlations']:
        print(f"{t['variable']:12s} lag {t['lag_days']:2d}d r={t['r']:+.3f} p={t['p']:.3f} holm={t['p_holm']:.3f}")
    print('kNN balanced acc:', r['knn']['balanced_accuracy'])


if __name__ == '__main__':
    main()
