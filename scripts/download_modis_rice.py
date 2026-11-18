"""Download MODIS MOD13Q1 NDVI (250 m, 16-day) and NASA POWER daily weather for one rice site.

Sources (both free, no login):
- ORNL DAAC MODIS/VIIRS Land Product Subsets web service, product MOD13Q1 (Collection 6.1):
  https://modis.ornl.gov/rst/api/v1/  -- max 10 composites per request, so we page.
- NASA POWER daily point API (MERRA-2 meteorology, ~0.5 x 0.625 deg grid):
  https://power.larc.nasa.gov/api/temporal/daily/point

Default site: An Giang / Dong Thap rice area near Long Xuyen (10.45 N, 105.45 E), +-20 km box.
The 2005-2008 brown planthopper / grassy & ragged stunt outbreak hit An Giang, Dong Thap and
Tien Giang. Change --lat/--lon/--km for another site.

Output (raw, git-ignored): data/raw/modis/<site>/{chunks/*.json, modis.npz, power.json}
Provenance (versioned):    data/field_provenance.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODIS = 'https://modis.ornl.gov/rst/api/v1/MOD13Q1'
POWER = 'https://power.larc.nasa.gov/api/temporal/daily/point'
BANDS = {'ndvi': '250m_16_days_NDVI', 'reliability': '250m_16_days_pixel_reliability'}
POWER_PARAMS = 'T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR'
AGENT = 'leaflab-student-client'


def get_json(url, tries=6):
    """GET JSON with retries; the ORNL service returns transient 502s and empty bodies."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode('utf-8')
            value = json.loads(body)
            if isinstance(value, str):  # service reports errors as a bare JSON string
                raise RuntimeError(value)
            return value
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ConnectionError) as e:
            wait = 5*(attempt+1)
            print(f'  retry {attempt+1}/{tries} in {wait}s: {type(e).__name__}', flush=True)
            time.sleep(wait)
    raise RuntimeError(f'failed after {tries} tries: {url}')


def modis_dates(lat, lon, start, end):
    dates = get_json(f'{MODIS}/dates?latitude={lat}&longitude={lon}')['dates']
    return [d for d in dates if start <= d['calendar_date'] <= end]


def download_band(lat, lon, km, band, dates, chunk_dir):
    """Fetch one band in 10-date pages, caching each page so reruns resume."""
    pages, header = [], None
    for i in range(0, len(dates), 10):
        page = dates[i:i+10]
        cache = chunk_dir/f"{band}_{page[0]['modis_date']}.json"
        if cache.exists():
            value = json.loads(cache.read_text(encoding='utf-8'))
        else:
            url = (f'{MODIS}/subset?latitude={lat}&longitude={lon}&band={band}'
                   f"&startDate={page[0]['modis_date']}&endDate={page[-1]['modis_date']}"
                   f'&kmAboveBelow={km}&kmLeftRight={km}')
            value = get_json(url)
            cache.write_text(json.dumps(value), encoding='utf-8')
        print(f"  {band} {page[0]['calendar_date']}..{page[-1]['calendar_date']}"
              f" ({len(value['subset'])} composites)", flush=True)
        header = header or {k: v for k, v in value.items() if k != 'subset'}
        pages.extend(value['subset'])
    return header, pages


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', default='angiang')
    p.add_argument('--lat', type=float, default=10.45)
    p.add_argument('--lon', type=float, default=105.45)
    p.add_argument('--km', type=int, default=20, help='half-width of the box (service max 100)')
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end', default='2010-12-31')
    p.add_argument('--out', type=Path, default=ROOT/'data'/'raw'/'modis')
    a = p.parse_args()

    out = a.out/a.site
    (out/'chunks').mkdir(parents=True, exist_ok=True)
    dates = modis_dates(a.lat, a.lon, a.start, a.end)
    print(f'{len(dates)} MOD13Q1 composites {dates[0]["calendar_date"]}..{dates[-1]["calendar_date"]}')

    arrays, headers, proc = {}, {}, {}
    for key, band in BANDS.items():
        header, pages = download_band(a.lat, a.lon, a.km, band, dates, out/'chunks')
        by_date = {s['calendar_date']: s for s in pages}
        missing = [d['calendar_date'] for d in dates if d['calendar_date'] not in by_date]
        if missing:
            sys.exit(f'{band}: service returned no data for {missing[:5]}...')
        h, w = int(header['nrows']), int(header['ncols'])
        arrays[key] = np.stack([np.asarray(by_date[d['calendar_date']]['data'], np.int16).reshape(h, w)
                                for d in dates])
        headers[key] = header
        proc[key] = sorted({s['proc_date'] for s in pages})
    np.savez_compressed(out/'modis.npz', dates=np.array([d['calendar_date'] for d in dates]),
                        ndvi_raw=arrays['ndvi'], reliability=arrays['reliability'].astype(np.int8),
                        cellsize=float(headers['ndvi']['cellsize']),
                        xllcorner=float(headers['ndvi']['xllcorner']),
                        yllcorner=float(headers['ndvi']['yllcorner']))

    s, e = a.start.replace('-', ''), a.end.replace('-', '')
    power_url = (f'{POWER}?parameters={POWER_PARAMS}&community=AG&longitude={a.lon}'
                 f'&latitude={a.lat}&start={s}&end={e}&format=JSON')
    power = get_json(power_url)
    (out/'power.json').write_text(json.dumps(power), encoding='utf-8')

    prov_path = ROOT/'data'/'field_provenance.json'
    prov = json.loads(prov_path.read_text(encoding='utf-8')) if prov_path.exists() else {}
    prov[a.site] = {
        'downloaded_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'lat': a.lat, 'lon': a.lon, 'half_width_km': a.km, 'start': a.start, 'end': a.end,
        'modis': {'product': 'MOD13Q1 (ORNL DAAC Land Product Subsets)', 'service': MODIS,
                  'bands': BANDS, 'n_composites': len(dates), 'grid': [int(headers['ndvi']['nrows']),
                  int(headers['ndvi']['ncols'])], 'cellsize_m': float(headers['ndvi']['cellsize']),
                  'tile': pages[0]['tile'], 'proc_dates': proc,
                  'example_request': headers['ndvi'].get('header')},
        'power': {'service': POWER, 'parameters': POWER_PARAMS.split(','),
                  'api_version': power['header']['api']['version'], 'sources': power['header']['sources'],
                  'note': 'POWER data can be revised; numbers are tied to the download date above.'},
    }
    prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out/"modis.npz"}, {out/"power.json"}, {prov_path}')


if __name__ == '__main__':
    main()
