# LeafLab v3 · Bài tập lớn Xử lý ảnh

Pipeline cổ điển cho ảnh lá lúa/cà phê. Lõi tự cài bằng NumPy; không dùng CNN hoặc bộ tách nền học sâu. Phân lớp bằng k-NN, k-NN sau phép chiếu LDA hoặc ridge nhân RBF, đều tự cài bằng NumPy và được chọn bằng kiểm định chéo cho từng cây; SVM/PCA và GrabCut là phương án thư viện đối chứng có ghi rõ.

## Chạy trên Windows

Yêu cầu Python 3.10+ (khuyến nghị 3.12), không cần GPU. Mở PowerShell tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/download_rice.py
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Nếu đã có thư viện và model v2, bấm `START_DEMO.bat`. Camera cần quyền trình duyệt và HTTPS/localhost; tải ảnh nếu truy cập HTTP nội bộ không mở được camera. Script tải UCI gộp theo ID, giữ group/split/mask đã chỉnh và tạo bản sao CSV trước khi ghi; không còn ghi đè toàn bộ manifest.

## Luồng xử lý

1. Đọc RGB, chuẩn hướng EXIF, cạnh dài tối đa 256 pixel.
2. Stage 0: gray-world có giới hạn gain, CLAHE trên L*, median tùy chọn. Mặt nạ chói lấy từ ảnh gốc. Mask lá vẫn dùng RGB gốc để giữ quy ước màu giữa các ablation.
3. Tách lá bằng hybrid ExG/HSV, HSV, ExG+Otsu hoặc ExG−ExR. Hình thái hỗ trợ SE chữ nhật. Đệm edge trước hình thái để không xóa thông tin lá chạm biên.
4. K-means a*b*. Cụm đông nhất được giả định là mô khỏe; chọn cụm xa nhất nếu khoảng cách ≥8. Luật Lab cũ và Otsu a* được giữ để đối chiếu. Lọc vùng dưới 3 pixel, đóng hình thái tùy chọn tắt mặc định.
5. Tính tỷ lệ vùng nghi tổn thương trên phần lá nhìn thấy. Pixel chói không tham gia phân cụm/đặc trưng; vùng chói nhiều phát cảnh báo vì tỷ lệ có thể bị đánh giá thấp.
6. Trích 403 đặc trưng. Nhóm vùng tổn thương 135: màu 102 (histogram HSV/Lab và moment Lab), kết cấu 26 (GLCM bốn hướng và LBP vuông bán kính 1,2), hình dạng 7; ROI dưới 1% lá trả 0 cho riêng nhóm này. Nhóm toàn lá 268, không phụ thuộc mặt nạ tổn thương: histogram hai chiều a*b* 10×10 trên ảnh đã tiền xử lý, và trên ảnh gốc theo ba kênh L*, a*, b*: LBP bán kính 1,2,3, histogram độ lớn gradient, GLCM khoảng cách 1,2,4. Giá trị histogram của nhóm toàn lá được lấy căn bậc hai.
7. Chuẩn hóa theo train, rồi một trong ba bộ phân lớp NumPy: k-NN, LDA có co ma trận + k-NN, hoặc ridge nhân RBF một-chọi-phần-còn-lại. Từ chối theo phiếu láng giềng/khoảng cách; với LDA, bán kính từ chối tính trên các mẫu để ngoài khi fit LDA. Demo có láng giềng gần nhất và các thành phần khoảng cách để giải thích; đây không phải độ quan trọng nhân quả.

`FEATURE_NAMES`, `FEATURE_DIM`, `FEATURE_GROUPS` trong `src/features.py` là nguồn chung cho app, model, thực nghiệm và báo cáo. Model dùng schema `leaflab-3`; model cũ bị từ chối với hướng dẫn tạo lại, không âm thầm chạy với vector sai chiều.

## Thực nghiệm và tái lập

```powershell
python -m pytest -q
python scripts/verify_algorithms.py
python run_experiments.py --manifest data/manifest.csv --out results
python scripts/compare_leaf_methods.py
```

- k-NN: k=1,3,5,7 × color/texture/shape/lesion/leaf/all, cùng folds phân nhóm, scaler train-only.
- LDA + k-NN trên leaf/all: shrinkage 0.03, 0.1, 0.3; k=1,5,11,21. Ridge nhân RBF trên leaf/all: gamma 0.1, 0.3, 1 (chia cho số đặc trưng); alpha 0.01, 0.1, 1. Cấu hình có macro-F1 CV cao nhất được huấn luyện lại và chấm trên test.
- SVM RBF: C=0.1,1,10; gamma=scale/0.01; PCA tắt hoặc giữ 95% phương sai. Scaler/PCA fit bên trong mỗi fold.
- Stress: RGB ×0.6/×1.4, Gaussian sigma 1/2, xoay 90/180/15 độ. Giữ model cố định, báo lỗi và accuracy end-to-end đầy đủ mẫu số.
- So sánh tiền xử lý và luật Lab cũ trên train/CV, không chọn tham số từ test.
- So sánh tách lá: hybrid, HSV, ExG+Otsu, ExG−ExR, SE 1×7/7×1, GrabCut OpenCV đối chứng. Minh họa định tính lấy TRAIN trong `leaf_methods_train.png`.
- So sánh tổn thương: a*b*, luật Lab cũ, Otsu a*; K=2,3,4. IoU chỉ xuất khi có mask tay segtest. Chọn ngưỡng bằng segval trước khi khóa test.

`results/metrics.json` là kết quả hiện tại; `cross_validation.json` là lưới k-NN; `predictions.csv` ghi từng ảnh; `verification.json` là đối chiếu số; `environment.json` ghi môi trường. Kết quả v1 lưu ở `results/baseline_v1/`, kết quả v2 (135 đặc trưng, k-NN) ở `results/baseline_leaflab-2/`. Không xem việc thêm bước mới là bảo đảm tăng accuracy. Xem [trạng thái sửa và giới hạn](docs/V2_CHANGELOG.md).

## Kết quả v3 trên test (results/metrics.json)

| Cây | Cấu hình được CV chọn | Accuracy | Macro-F1 | v2 (accuracy) |
|---|---|---:|---:|---:|
| Cà phê (335 ảnh) | leaf 268, LDA 0.1 + k-NN k=21 | 88,4% | 0,841 | 63,9% |
| Lúa (30 ảnh) | leaf 268, ridge RBF gamma 0.1, alpha 0.01 | 90,0% | 0,900 | 63,3% |

Giới hạn cần nêu khi dùng các số này:

- Nhóm chia theo ảnh vì nguồn không có ID cây/phiên chụp. Ở BRACOL, nhãn gắn với thứ tự chụp và riêng màu nền đã đoán được nhãn khoảng 69% (kiểm tra ngoài repo); khi giữ các ảnh có số thứ tự liền nhau cùng một phía, accuracy cà phê còn khoảng 75–81%. Độ chính xác ngoài thực tế chưa đo được khi chưa có ảnh tự chụp.
- Tập test lúa chỉ 30 ảnh (mỗi ảnh 3,3 điểm) và đã được xem nhiều lần trong quá trình phát triển v3; chênh lệch CV giữa các bộ phân lớp nằm trong độ nhiễu.
- Cercospora vẫn yếu (16/34). Accuracy cà phê còn 65–68% khi đổi độ sáng ×0,6/×1,4 và 44% khi mờ sigma 2.
- Tên file `models/<cây>_knn.npz` được giữ dù mô hình lúa hiện là ridge nhân RBF; trường `method` trong file cho biết loại mô hình.

`data/samples/` chứa 24 ảnh thuộc split test (3 ảnh mỗi lớp) để mục “Ảnh mẫu” hoạt động khi triển khai ở nơi không có `data/raw/` (thư mục này không được đưa lên Git).

## Mức độ và khuyến nghị

`data/severity_rules.json` chứa ngưỡng quy ước diện tích <1%, 1–5%, 5–20%, ≥20% và bảng khuyến nghị bệnh × mức độ. Mức thấp tập trung theo dõi, mức cao yêu cầu khảo sát thêm và xác nhận chuyên môn; trường hợp từ chối có hướng dẫn riêng.

**IRRI chưa được ánh xạ:** nguồn PDF chính thức không truy cập được trong phiên xác minh; bản trích tìm kiếm có hàng bảng không nhất quán nên không dùng làm chuẩn. `irri_score=null` và trạng thái chờ xác minh được hiển thị rõ. Không dùng một thang 0–9 chung cho mọi bệnh hoặc cho cà phê, không biến tỷ lệ vùng nghi tổn thương thành chỉ định phun thuốc.

## Dữ liệu cần hoàn thiện

UCI Rice có 120 ảnh, 3 bệnh, không có lá khỏe và không có mask pixel. BRACOL (cà phê) dùng 1.342 ảnh toàn lá, 5 lớp. Chưa có ≥100 ảnh tự chụp và ≥30 mask tay; trạng thái tiếp tục `pending`. Không dùng ảnh tổng hợp để hoàn thành các chỉ tiêu này. Dữ liệu NASA không được dùng trong lần chạy này.

Xem [hướng dẫn dữ liệu](data/README.md), [công cụ đa giác](docs/annotate.html) và [lộ trình 8 bước](docs/HUONG_DAN.md). Nhập ảnh field bắt buộc có `--group-map` chứa ID cây/lá/phiên và split thật; không coi mỗi ảnh chụp liên tiếp là mẫu độc lập.

## Tài liệu và cấu trúc

- `src/algorithms.py`: thuật toán NumPy; `src/pipeline.py`: cấu hình và chuỗi xử lý.
- `src/features.py`: đặc trưng ROI/hình dạng và schema dùng chung.
- `src/model.py`: k-NN, kiểm tra model, giải thích khoảng cách.
- `src/advice.py`: mức độ và bảng khuyến nghị; `src/data.py`: manifest và gộp an toàn.
- `scripts/`: tải/nhập ảnh, nhãn tay, phương án thư viện đối chứng, kiểm chứng và tạo báo cáo.
- `docs/BAO_CAO_BTL_LeafLab.docx` và `.pdf`: báo cáo học thuật, số đo lấy từ JSON hiện tại.
- `docs/BAO_VE.md`: kịch bản và đề cương 18 slide, chưa phải PowerPoint/video.

Tạo lại báo cáo: cài `requirements-docs.txt`, chạy `python scripts/build_report.py`, rồi `python scripts/report_pdf_fallback.py` trên Windows có font Times/Segoe UI Symbol. Kiểm tra phân trang sau khi chỉnh Word. Tệp PDF được dựng từ cùng nội dung; Word giữ công thức Office Math chỉnh sửa được.

Nguồn và giấy phép ở [SOURCES.md](docs/SOURCES.md). Nhóm cần đọc, tự chạy, giải thích mã và khai báo AI; điền tên, đóng góp, nhật ký, repo GitHub và chữ ký thật. Không tạo lịch sử commit hay minh chứng thực địa giả.
