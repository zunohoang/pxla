"""Streamlit page: field-scale rice stress from MODIS + NASA POWER (results of run_field.py)."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.remote import (FieldConfig, decode_ndvi, loyo_zscore, rice_mask, severity_label,  # noqa: E402
                        slot_of, stress_series)

RESULTS = ROOT/'results'/'field'
SITE = ROOT/'data'/'raw'/'modis'/'angiang'
VAR_VI = {'RH2M': 'Độ ẩm', 'T2M': 'Nhiệt độ', 'PRECTOTCORR': 'Tổng mưa', 'WET_DAYS': 'Số ngày mưa'}

st.set_page_config(page_title='LeafLab | Ruộng lúa từ vệ tinh', page_icon='🛰️', layout='wide')
st.title('🛰️ Ruộng lúa từ vệ tinh (MODIS + NASA POWER)')
st.caption('An Giang / Đồng Tháp, 2000-2010. Pixel 250 m (~5,4 ha) không thấy được vết bệnh trên lá: '
           'đây là **bất thường sinh trưởng** có thể do bệnh, hạn, lũ hoặc lệch lịch thời vụ, không phải chẩn đoán bệnh.')

path = RESULTS/'field_analysis.json'
if not path.exists():
    st.warning('Chưa có kết quả. Chạy `python scripts/download_modis_rice.py` rồi `python run_field.py`.')
    st.stop()
r = json.loads(path.read_text(encoding='utf-8'))

c = st.columns(4)
c[0].metric('Diện tích lúa', f"{r['rice_mask']['area_km2']:.0f} km²")
c[1].metric('Ảnh dùng được', f"{r['composites_usable']}/{r['n_composites']}",
            help='Ảnh có <30 % pixel lúa không mây bị loại nhưng vẫn được đếm.')
o = r['outbreak_test']
c[2].metric('Bất thường 2006-07 vs năm khác', f"{o['observed']*100:.1f}% vs {o['others_mean']*100:.1f}%",
            help=f"Kiểm định hoán vị chính xác: hạng {o['rank_of_target']}/{o['n_sets']}, p = {o['p_one_sided']:.2f}")
k = r['knn']['balanced_accuracy']
c[3].metric('k-NN thời tiết (balanced acc.)', f"{k['knn_weather']:.2f}",
            delta=f"{k['knn_weather']-k['majority']:+.2f} so với đoán 'thấp'", delta_color='normal')

tabs = st.tabs(['1 · Mặt nạ lúa', '2 · Xem từng ảnh', '3 · Chuỗi thời gian', '4 · Thời tiết', '5 · Hạn chế'])

with tabs[0]:
    st.image(str(RESULTS/'field_rice_mask.png'), width='stretch')
    st.write(f"Ngưỡng Otsu trên biên độ mùa vụ (p90-p10 NDVI) = **{r['rice_mask']['amplitude_threshold']:.2f}**. "
             'Lúa ngập nước rồi xanh lại 2-3 lần/năm nên biên độ lớn; sông, đô thị, vườn cây có biên độ nhỏ.')


@st.cache_data(show_spinner='Đang tính z-score từ dữ liệu MODIS...')
def load_series():
    d = np.load(SITE/'modis.npz')
    cfg = FieldConfig()
    dates = [str(x) for x in d['dates']]
    ndvi = decode_ndvi(d['ndvi_raw'], d['reliability'], cfg)
    slots, years = slot_of(dates)
    rice, _ = rice_mask(ndvi, cfg)
    z = loyo_zscore(ndvi, slots, years, cfg)
    frac, cover, masks = stress_series(z, rice, cfg)
    return dates, ndvi, z, rice, frac, cover, masks


def colour(x, lo, hi, cmap):
    import matplotlib
    v = np.clip((x-lo)/(hi-lo), 0, 1)
    rgb = (matplotlib.colormaps[cmap](np.nan_to_num(v))[..., :3]*255).astype(np.uint8)
    rgb[~np.isfinite(x)] = 200  # grey = cloud / no data
    return rgb


with tabs[1]:
    if not (SITE/'modis.npz').exists():
        st.info('Không có dữ liệu thô (data/raw/modis/angiang/modis.npz) nên không xem được từng ảnh.')
    else:
        dates, ndvi, z, rice, frac, cover, masks = load_series()
        i = st.select_slider('Ảnh tổng hợp 16 ngày', options=list(range(len(dates))),
                             value=dates.index('2007-01-01') if '2007-01-01' in dates else 0,
                             format_func=lambda j: dates[j])
        f = frac[i]
        st.write(f"Không mây: **{cover[i]*100:.0f}%** pixel lúa · Bất thường: "
                 f"**{'—' if np.isnan(f) else f'{f*100:.1f}%'}** · Mức: **{severity_label(f)}**")
        a, b, cc = st.columns(3)
        a.image(colour(ndvi[i], 0, .9, 'YlGn'), caption='NDVI (xám = mây)', width='stretch')
        b.image(colour(np.where(rice, z[i], np.nan), -3, 3, 'RdBu'),
                caption='z-score so với cùng kỳ các năm khác (đỏ = thấp)', width='stretch')
        over = np.full(rice.shape+(3,), 235, np.uint8)
        over[rice] = (120, 170, 120)
        over[masks[i]] = (200, 40, 40)
        cc.image(over, caption=f'Vùng bất thường (z ≤ {FieldConfig().z_threshold}, ≥ {FieldConfig().min_area_px} pixel)',
                 width='stretch')

with tabs[2]:
    st.image(str(RESULTS/'field_stress_timeseries.png'), width='stretch')
    st.image(str(RESULTS/'field_stress_annual.png'), width='stretch')
    st.image(str(RESULTS/'field_stress_maps.png'), width='stretch')
    df = pd.DataFrame([{'Năm': y['year'], 'Ảnh dùng được': f"{y['usable']}/{y['composites']}",
                        'Bất thường TB (%)': None if y['mean_stressed_fraction'] is None else round(y['mean_stressed_fraction']*100, 1),
                        'Cao nhất (%)': None if y['max_stressed_fraction'] is None else round(y['max_stressed_fraction']*100, 1),
                        **{s: y['severity_counts'][s] for s in ('none', 'light', 'moderate', 'severe', 'no_data')}}
                       for y in r['per_year']])
    st.dataframe(df, hide_index=True, width='stretch')

with tabs[3]:
    st.image(str(RESULTS/'field_weather_lags.png'), width='stretch')
    st.dataframe(pd.DataFrame([{'Biến': VAR_VI[t['variable']], 'Trễ (ngày)': t['lag_days'], 'Spearman r': round(t['r'], 3),
                                'p': round(t['p'], 3), 'p (Holm)': round(t['p_holm'], 3), 'n': t['n']}
                               for t in r['correlations']]), hide_index=True, width='stretch')
    st.write('**k-NN** (k = {k}, giữ lại từng năm làm tập kiểm tra), đặc trưng: {f}'.format(
        k=r['knn']['k'], f=', '.join(r['knn']['features'])))
    st.dataframe(pd.DataFrame([{'Phương pháp': n, 'Balanced accuracy': round(v, 3)} for n, v in k.items()]),
                 hide_index=True)

with tabs[4]:
    for line in r['caveats']:
        st.markdown(f'- {line}')
    st.markdown('- Tham số trong `FieldConfig` cố định trước khi xem kết quả; không chỉnh theo năm dịch.')
