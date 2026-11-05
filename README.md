# LeafLab · Bài tập lớn Xử lý ảnh

Chẩn đoán hỗ trợ bệnh lá lúa/cà phê từ ảnh điện thoại bằng xử lý ảnh truyền thống. Thuật toán lõi tự viết bằng NumPy; k-NN tự viết dùng cho demo, SVM thư viện dùng làm đối chiếu. Đề tài theo yêu cầu người dùng; quy cách theo file hướng dẫn BTL.

## Bắt đầu nhanh trên Windows

Yêu cầu Python 3.10 trở lên (khuyến nghị 3.12), không cần GPU. Mở PowerShell trong thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_rice.py
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ localhost mà Streamlit in ra. Nếu đã có thư viện và mô hình, chỉ cần lệnh cuối. Có thể bấm `START_DEMO.bat`. Mục **Ảnh mẫu** hoạt động sau khi tải UCI. Camera cần quyền trình duyệt; dùng tải ảnh nếu điện thoại truy cập qua HTTP nội bộ không cho mở camera.

## Làm bài theo 8 bước

1. Đọc [lộ trình thực hiện](docs/HUONG_DAN.md), thống nhất đề với giảng viên vì Đề 13 trong file gốc là làn đường.
2. Chạy dữ liệu UCI để hiểu pipeline và kiểm chứng môi trường.
3. Đọc `src/algorithms.py`, làm lại ví dụ tính tay và chạy `python scripts/verify_algorithms.py`.
4. Bổ sung ảnh cà phê theo [hướng dẫn dữ liệu](data/README.md); không trộn nhãn giữa hai cây.
5. Chụp ≥100 ảnh thật; chia theo cây/phiên chụp, gán nhãn lớp với người có chuyên môn.
6. Gán nhãn pixel ≥30 ảnh test bằng [công cụ đa giác](docs/annotate.html). Dùng tập segval riêng để điều chỉnh ngưỡng.
7. Chạy `python run_experiments.py` để tạo toàn bộ JSON, CSV, biểu đồ và mô hình. Lưu `manifest.csv` cùng kết quả để tái lập.
8. Cập nhật báo cáo bằng kết quả của nhóm; điền tên, đóng góp, nhật ký, liên kết GitHub thật và luyện bảo vệ theo `docs/BAO_VE.md`.

## Năm chức năng bắt buộc

| Chức năng | Cài đặt |
|---|---|
| Tách lá bằng ExG/HSV và hình thái | `src/algorithms.py`, `src/pipeline.py` |
| K-means trong Lab | `kmeans`, `lesion_mask` |
| Tỉ lệ diện tích và mức độ | `process`, mặt nạ tổn thương luôn nằm trong mặt nạ lá |
| Histogram, LBP, GLCM và k-NN | 74 đặc trưng; chuẩn hóa chỉ trên train; `src/model.py` |
| Demo chụp/tải, bệnh, mức độ, khuyến nghị | `app.py`, tải mask/JSON, cơ chế từ chối |

## Kết quả và tính trung thực

Kết quả thật ở `results/metrics.json`, từng ảnh ở `results/predictions.csv`, CV ở `results/cross_validation.json`, đối chiếu thư viện ở `results/verification.json`. `results/environment.json` ghi môi trường. Thời gian phụ thuộc máy và không hoàn toàn tái lập.

Bản chạy ban đầu sử dụng UCI, chưa chứng minh hiệu quả ngoài đồng. UCI không có lớp lá khỏe. Mô hình chỉ phân biệt các bệnh đã học; cơ chế từ chối theo phiếu/khoảng cách không bảo đảm phát hiện mọi ảnh lạ. Tỉ lệ vùng nghi tổn thương không phải kết luận tác nhân bệnh. Ngưỡng mức độ <1%, 1–5%, 5–20%, ≥20% là quy ước đồ án, chưa được chuyên gia hiệu chuẩn.

Chưa có ảnh tự chụp và mask thật thì `field_requirement`/`segmentation.status` là `pending`, không xuất IoU giả. Chưa nhập cà phê thì demo vẫn phân đoạn nhưng không dự đoán tên bệnh cà phê. Để so sánh nền trơn với ngoài đồng, cần báo cáo số ảnh, phân bố lớp và dùng cùng taxonomy; mỗi miền không có lớp nào đó thì ghi rõ macro-F1 vẫn tính trên danh sách lớp cố định.

Ảnh test lỗi tách lá được tính là lỗi phân lớp với cột `processing_failed`, giữ nguyên mẫu số toàn tập. Lỗi ảnh train dừng thí nghiệm để rà soát trước khi học. Stress độ sáng báo riêng số lỗi và chỉ số trên ảnh xử lý thành công. Hãy đọc đúng mẫu số khi so sánh.

## Kiểm thử và tái lập

```powershell
python -m pytest -q
python scripts/verify_algorithms.py
python run_experiments.py --manifest data/manifest.csv --out results
```

So sánh HSV/Lab với scikit-image; co/giãn với OpenCV cùng quy ước biên; GLCM với graycomatrix/graycoprops; K-means cùng tâm đầu với sklearn; k-NN cùng chuẩn hóa với sklearn. LBP của dự án dùng lân cận vuông, kiểm chứng độc lập ví dụ tính tay; không đánh đồng với nội suy tròn của scikit-image.

`requirements-lock.txt` lưu phiên bản đã chạy; cài bằng `pip install -r requirements-lock.txt` để tái tạo môi trường gần nhất. Seed 42 cố định dữ liệu, CV, khởi tạo và lấy mẫu K-means. Sau khi bổ sung dữ liệu chạy lại báo cáo, không sao chép kết quả UCI sang ảnh tự chụp.

## Báo cáo và tài liệu

- `docs/BAO_CAO_BTL_LeafLab.pdf`: bản báo cáo 34 trang, gồm 25 trang nội dung chính.
- `docs/BAO_CAO_BTL_LeafLab.docx`: bản chỉnh sửa được; công thức là đối tượng Office Math.
- `docs/HUONG_DAN.md`: lộ trình 8 bước, phân công, thu ảnh, đánh giá và hồ sơ nộp.
- `docs/BAO_VE.md`: đề cương 18 slide, bài nói, video và câu hỏi vấn đáp. Chưa phải tệp PowerPoint hay video.
- `docs/annotate.html`: công cụ đa giác chạy cục bộ để tạo nhãn tay.

Tạo lại bản thảo Word bằng `python scripts/build_report.py` sau khi có kết quả JSON. Cài thêm `requirements-docs.txt` nếu cần. Trên Windows, `python scripts/report_pdf_fallback.py` tạo PDF cùng nội dung bằng font Times và font ký hiệu toán của hệ thống. Bản PDF đã được kết xuất và kiểm tra; khi sửa Word, cần kiểm tra lại phân trang trong Word trước khi nộp. Script báo cáo hiện tập trung phân tích baseline UCI, nhóm phải viết bổ sung phần diễn giải khi có cà phê/ảnh thực địa.

## Cấu trúc

```text
app.py                     giao diện demo
run_experiments.py         CV, test, khảo sát, xuất kết quả
src/algorithms.py          thuật toán NumPy
src/pipeline.py            phân đoạn và đặc trưng
src/model.py               k-NN và lưu model NPZ
src/data.py                kiểm tra manifest và rò rỉ dữ liệu
src/evaluation.py          accuracy, macro-F1, IoU, Dice
scripts/                   tải dữ liệu, nhập ảnh, gán nhãn, kiểm chứng
tests/                     kiểm thử hành vi và giao diện
data/                      manifest, nguồn, hướng dẫn
models/                    mô hình đã huấn luyện
results/                   kết quả thực nghiệm
docs/                      báo cáo, lộ trình, bảo vệ, gán nhãn
```

## Ghi công và liêm chính

Xem tài liệu tham khảo trong báo cáo và [SOURCES.md](docs/SOURCES.md). Nhóm cần khai báo việc sử dụng AI, xác minh từng kết quả và hiểu từng thuật toán. Không tạo chữ ký, tên thành viên, ảnh thực địa, video bảo vệ hay lịch sử commit giả. Mã nguồn được tạo cho dự án này với hỗ trợ AI; các công thức/ý tưởng đã ghi nguồn, không sao chép một repo có sẵn.
