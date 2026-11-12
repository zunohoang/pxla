"""Streamlit phone capture/upload demo; run: streamlit run app.py."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.model import KNN
from src.pipeline import Config, load_rgb, process
from src.features import FEATURE_DIM
from src.advice import recommendation, severity

NAMES = {'bacterial_leaf_blight': 'Bạc lá lúa', 'brown_spot': 'Đốm nâu',
         'leaf_smut': 'Đốm than lá lúa', 'healthy': 'Lá khỏe',
         'rust': 'Gỉ sắt', 'leaf_miner': 'Sâu vẽ bùa',
         'cercospora': 'Đốm mắt cua'}


def png_bytes(array):
    """Encode downloadable RGB/mask PNG without a shared temporary file."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def main():
    """Show all required processing stages and abstain if classifier is absent."""
    st.set_page_config(page_title='LeafLab | Phân tích bệnh lá', page_icon='🌿', layout='wide')
    st.title('LeafLab · Phân tích bệnh lá từ ảnh')
    st.caption('Bài tập lớn Xử lý ảnh · NumPy · K-means Lab · Đặc trưng màu và kết cấu · k-NN')
    crop = st.sidebar.selectbox('Cây trồng', ['rice', 'coffee'],
                                format_func=lambda c: 'Lúa' if c == 'rice' else 'Cà phê')
    st.sidebar.markdown('**Cách chụp**\n\nMột lá, đủ sáng, rõ nét, nền tương phản. Giữ toàn bộ lá trong ảnh.')
    model_path = Path('models') / f'{crop}_knn.npz'
    model, config = None, Config()
    if model_path.is_file():
        try:
            model, config, saved_crop = KNN.load(model_path)
        except (ValueError, KeyError, OSError) as exc:
            st.error(f'Không nạp được mô hình: {exc}')
            st.info('Chạy python run_experiments.py để tạo lại mô hình tương thích.')
            st.stop()
        if saved_crop != crop:
            st.error('Mô hình không đúng cây trồng.')
            st.stop()
        st.sidebar.success('Đã nạp mô hình ' + crop)
        st.sidebar.write('Các lớp đã học: ' + ', '.join(NAMES.get(c, c) for c in model.classes))
        if 'healthy' not in model.classes:
            st.sidebar.warning('Mô hình chưa có lớp lá khỏe. Không dùng để kết luận lá khỏe hay bệnh.')
    else:
        st.sidebar.info('Chưa có mô hình cây này. Vẫn có thể xem phân đoạn và diện tích.')
    mode = st.radio('Nguồn ảnh', ['Tải ảnh', 'Chụp ảnh', 'Ảnh mẫu'], horizontal=True)
    source = None
    if mode == 'Tải ảnh':
        source = st.file_uploader('Ảnh lá JPG hoặc PNG', type=['jpg', 'jpeg', 'png'])
    elif mode == 'Chụp ảnh':
        source = st.camera_input('Chụp một lá')
        st.caption('Trình duyệt cần cấp quyền camera. Điện thoại truy cập camera qua HTTPS hoặc localhost.')
    else:
        # data/raw is not versioned (large); data/samples holds a few test-split images for deployed demos.
        examples = []
        for root in [f'data/raw/{crop}', f'data/samples/{crop}']:
            examples = sorted(p for p in Path(root).glob('*/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            if examples:
                break
        if examples:
            source = st.selectbox('Chọn ảnh', examples, format_func=lambda p: p.parent.name + ' / ' + p.name)
        else:
            st.info('Chưa có ảnh mẫu; xem hướng dẫn tải dữ liệu trong README.')
    if source is None:
        st.info('Chọn ảnh để xem từng bước xử lý và kết quả.')
        return
    try:
        rgb = load_rgb(source, config.max_side)
        with st.spinner('Đang tách lá và phân tích tổn thương…'):
            result = process(rgb, config)
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        st.error(str(exc))
        return
    for warning in result['warnings']:
        st.warning(warning)
    label, detail = 'Chưa có mô hình phân lớp', None
    if model is not None:
        detail = model.predict_details(result['features'][None])[0]
        label = 'Chưa đủ bằng chứng để nhận dạng' if detail['rejected'] else NAMES.get(detail['label'], detail['label'])
    a, b, c = st.columns(3)
    a.metric('Loại bệnh dự đoán', label)
    b.metric('Diện tích nghi tổn thương', f"{result['ratio']*100:.2f}%")
    c.metric('Mức độ theo quy ước đồ án', result['severity'])
    st.caption('Tỉ lệ trên diện tích chiếu của phần lá nhìn thấy. Ngưỡng <1%, 1–5%, 5–20%, ≥20% là quy ước kỹ thuật, chưa được hiệu chuẩn nông học.')
    if detail:
        st.write(f"Tỉ lệ phiếu láng giềng: {detail['vote']:.0%}. Đây không phải xác suất bệnh.")
    advice = recommendation(detail['label'] if detail else None, result['ratio'],
                            rejected=detail['rejected'] if detail else True)
    st.info('Khuyến nghị: '+advice)
    st.caption('IRRI: chưa ánh xạ điểm vì bảng theo từng bệnh chưa được xác minh. Mức trên là quy ước diện tích của đồ án.')
    tabs = st.tabs(['1 · Ảnh đầu vào', '2 · Tách lá', '3 · Vết nghi bệnh', '4 · Giải thích', '5 · Tiền xử lý'])
    with tabs[0]:
        st.image(rgb, width=600)
    with tabs[1]:
        st.image(result['leaf'].astype('uint8')*255, width=600)
        st.download_button('Tải mặt nạ lá', png_bytes(result['leaf'].astype('uint8')*255), 'leaf_mask.png', 'image/png')
    with tabs[2]:
        st.image(result['overlay'], width=600)
        st.download_button('Tải mặt nạ vết nghi bệnh', png_bytes(result['lesion'].astype('uint8')*255), 'lesion_mask.png', 'image/png')
    with tabs[3]:
        st.write(f'RGB gốc → ExG/HSV → mask lá. Ảnh chuẩn hóa + mask lá → K-means a*b* → vùng tổn thương → {FEATURE_DIM} đặc trưng (vùng tổn thương và toàn lá) → chuẩn hóa, LDA nếu mô hình có dùng → k-NN.')
        st.dataframe(pd.DataFrame(result['centers'], columns=['L*', 'a*', 'b*']))
        st.json(config.to_dict())
        if model is not None:
            explanation = model.explain(result['features'])
            st.write('Các mẫu tham chiếu gần nhất')
            st.dataframe(explanation['neighbors'])
            st.write('Các thành phần đóng góp lớn nhất vào khoảng cách tới mẫu gần nhất')
            st.dataframe(explanation['distance_terms'])
            st.caption('Đây là thành phần khoảng cách, không phải bằng chứng nhân quả hoặc độ quan trọng của bệnh.')
    with tabs[4]:
        st.image(result['processed'], width=600, caption='Gray-world và CLAHE trên L* theo cấu hình đã lưu')
        st.image(result['glare'].astype('uint8')*255, width=600, caption='Mặt nạ chói; chỉ phần chói nằm trong lá bị loại khỏi phân tích tổn thương')
    record = dict(crop=crop, prediction=detail, visible_lesion_ratio=result['ratio'],
                  severity=severity(result['ratio'], crop, detail['label'] if detail else None),
                  recommendation=advice, warnings=result['warnings'], config=config.to_dict())
    st.download_button('Tải kết quả JSON', json.dumps(record, ensure_ascii=False, indent=2),
                       'leaflab_result.json', 'application/json')


if __name__ == '__main__':
    main()
