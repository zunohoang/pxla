"""Streamlit phone capture/upload demo; run: streamlit run app.py."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

from src.model import KNN
from src.pipeline import Config, load_rgb, process

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
        model, config, saved_crop = KNN.load(model_path)
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
        examples = sorted(Path(f'data/raw/{crop}').glob('*/*'))
        examples = [p for p in examples if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
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
    st.info('Khuyến nghị: chụp thêm cả hai mặt lá và toàn cây, theo dõi sự lan rộng, giữ ảnh đối chiếu và nhờ cán bộ bảo vệ thực vật xác nhận trước khi xử lý.')
    tabs = st.tabs(['1 · Ảnh đầu vào', '2 · Tách lá', '3 · Vết nghi bệnh', '4 · Giải thích'])
    with tabs[0]:
        st.image(rgb, width=600)
    with tabs[1]:
        st.image(result['leaf'].astype('uint8')*255, width=600)
        st.download_button('Tải mặt nạ lá', png_bytes(result['leaf'].astype('uint8')*255), 'leaf_mask.png', 'image/png')
    with tabs[2]:
        st.image(result['overlay'], width=600)
        st.download_button('Tải mặt nạ vết nghi bệnh', png_bytes(result['lesion'].astype('uint8')*255), 'lesion_mask.png', 'image/png')
    with tabs[3]:
        st.write('ExG/HSV → mở và đóng → thành phần liên thông → Lab → K-means → chọn cụm theo màu → diện tích → histogram/LBP/GLCM → k-NN.')
        st.dataframe(result['centers'], column_config={0: 'L*', 1: 'a*', 2: 'b*'})
        st.json(config.to_dict())
    record = dict(crop=crop, prediction=detail, visible_lesion_ratio=result['ratio'],
                  severity=result['severity'], warnings=result['warnings'], config=config.to_dict())
    st.download_button('Tải kết quả JSON', json.dumps(record, ensure_ascii=False, indent=2),
                       'leaflab_result.json', 'application/json')


if __name__ == '__main__':
    main()
