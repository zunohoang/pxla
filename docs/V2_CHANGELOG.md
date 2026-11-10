# Đối chiếu bản sửa LeafLab v2

Ngày kiểm tra: 05/11/2026. Phạm vi: xử lý các vấn đề trong bản “So sánh bản thiết kế với repo LeafLab” do người dùng gửi. Đây là bản sửa kỹ thuật; các yêu cầu thu thập dữ liệu thật vẫn phải hoàn thiện trước khi nộp.

## 1. Các thay đổi đã triển khai

| Nhóm góp ý | Thay đổi | Minh chứng |
|---|---|---|
| Tiền xử lý | Gray-world có giới hạn gain, CLAHE trên L*, chuyển Lab về RGB; median tùy chọn, mask chói | `src/algorithms.py`, `src/pipeline.py`, `results/verification.json` |
| Chọn vùng bệnh | K-means a*b*, cụm đông nhất được giả định khỏe; cụm xa nhất chỉ được chọn khi khoảng cách ≥8; bỏ vùng dưới 3 pixel; đóng hình thái tùy chọn | `src/pipeline.py`; giữ `lab_rule` và `otsu_a` làm đối chứng |
| Đặc trưng | Màu và kết cấu chỉ lấy trong ROI tổn thương không chói; bỏ góc phần tư; thêm hình dạng; 135 chiều = màu 102 + kết cấu 26 + hình dạng 7 | `src/features.py`; LBP vuông bán kính 1/2, GLCM trung bình bốn hướng |
| Schema mô hình | Hằng số tên/nhóm/chiều dùng chung, `leaflab-2`, kiểm tra khi nạp; app hướng dẫn tạo lại model cũ | `src/model.py`, model NPZ đã chạy lại |
| Mức độ và khuyến nghị | Tách bảng ngưỡng và lời khuyên khỏi pipeline; lời khuyên đổi theo bệnh và mức độ, có trường hợp từ chối | `data/severity_rules.json`, `src/advice.py`, test tại ngưỡng biên |
| Tách lá đối chứng | Hybrid, HSV, ExG+Otsu, ExG−ExR, SE 1×7/7×1; GrabCut OpenCV được ghi rõ là thư viện đối chứng | `scripts/reference_methods.py`, `results/leaf_methods_train.png` |
| Phân lớp | 16 cấu hình k-NN qua CV, thêm SVM RBF với C/gamma/PCA trong fold; scaler chỉ fit train | `run_experiments.py`, `results/cross_validation.json`, `results/metrics.json` |
| Độ bền | Stress sáng ×0,6/×1,4; Gaussian sigma 1/2; xoay 90°/180°/15°; báo đầy đủ lỗi | `crops.rice.robustness_stress` trong JSON |
| Diễn giải | Hiện ID/nhãn/khoảng cách láng giềng và các thành phần khoảng cách lớn nhất | Tab giải thích; không gọi đây là giải thích nhân quả |
| An toàn manifest | Gộp theo ID và sao lưu trước ghi; giữ group/split/mask đã chỉnh; ảnh field phải có group-map | `src/data.py`, script tải/nhập, `data/group_map_template.csv` |

Mask lá vẫn tính từ RGB gốc để giữ cùng mẫu số khi ablation tiền xử lý. Ảnh đã chuẩn hóa dùng cho K-means và đặc trưng. Pixel chói bị loại khỏi tử số/ROI nhưng không bị trừ khỏi diện tích lá; vì vậy ảnh chói có thể bị đánh giá thiếu tổn thương. Demo có cảnh báo khi chói chiếm trên 3% lá.

Các ngưỡng khoảng cách 8, vùng nhỏ 3 pixel và diện tích 1/5/20% là mặc định kỹ thuật chưa hiệu chuẩn bằng nhãn tay. Cụm đông nhất không luôn là mô khỏe. ROI dưới 1% trả vector 0 theo thiết kế, có thể làm mất thông tin ở lá ít triệu chứng. Không tuyên bố các thay đổi này đã làm mask đúng hơn khi chưa có ground truth.

## 2. Lỗi đánh giá được sửa

- Không tách được lá khi đánh giá mask: giữ ảnh trong mẫu số, ghi nguyên nhân; IoU/Dice/leaf-IoU bằng 0, kể cả khi GT tổn thương rỗng.
- Mask lá GT rỗng, mask bệnh ngoài lá hoặc sai kích thước: từ chối dữ liệu không hợp lệ.
- Hình thái có đệm trước khi crop lại: giữ vùng lá chạm biên để cảnh báo hoạt động đúng.
- Script tải lại không ghi đè phần gán nhãn và phân chia tập đã sửa trong manifest.
- Mô hình schema cũ không bị dùng nhầm với vector đặc trưng mới.
- Bộ chấm CV SVM gọi trực tiếp macro-F1 đa lớp theo danh sách lớp cố định, tránh lỗi `pos_label` của scorer với nhãn chuỗi.

## 3. Kết quả chạy lại, không chỉ chọn số tăng

Giao thức giữ nguyên 120 ảnh UCI: 90 train, 30 test; seed 42. Chọn nhóm đặc trưng và k bằng CV trong train. Model v2 trích đủ 135 chiều nhưng cấu hình CV chọn **color 102 chiều, k=1** cho demo. Điều này không có nghĩa nhóm hình dạng chưa được cài đặt.

| Chỉ số | v1 lưu trữ | v2 hiện tại |
|---|---:|---:|
| k-NN accuracy test | 90,0% | 63,3% (19/30) |
| k-NN macro-F1 test | 0,9019 | 0,6284 |
| Accuracy stress ×0,6 | 36,7% | 40,0% |
| Accuracy stress ×1,4 | 50,0% | 50,0% |

SVM v2 đạt accuracy 56,7%, macro-F1 0,5783. Train/CV của SVM chọn PCA giữ 95% phương sai, C=0,1, gamma=scale. Lưới và tiêu chí hai mô hình được ghi trong báo cáo.

Ablation cố định k=3, toàn bộ đặc trưng: macro-F1 CV không tiền xử lý 0,6493; có tiền xử lý 0,6007; luật Lab cũ không tiền xử lý 0,6671. Không kết luận CLAHE cải thiện phân lớp từ những số này. Không đổi ngưỡng theo test để nâng số. So sánh v1/v2 thay đổi nhiều khối cùng lúc, không tách riêng tác động của từng khối.

Kết quả cũ: `results/baseline_v1/`. Kết quả hiện tại: `results/metrics.json`, dự đoán từng ảnh: `results/predictions.csv`. Độ chính xác giảm là giới hạn thực nghiệm của bản thiết kế mới và được giữ nguyên trong báo cáo.

## 4. Kiểm tra đã thực hiện

- `python -m pytest -q`: **21 passed**, gồm các kiểm thử Streamlit AppTest và lỗi hồi quy.
- `python scripts/verify_algorithms.py`: **22/22 đối chiếu đạt dung sai**, gồm CLAHE cả ảnh chia hết/không chia hết lưới ô, Lab hai chiều, median, signed GLCM, eccentricity, solidity, EDT, Otsu và các thuật toán cũ.
- `python run_experiments.py`: hoàn thành; model và JSON hiện tại cùng schema `leaflab-2`.
- `python scripts/compare_leaf_methods.py`: ảnh minh họa TRAIN; chưa gán điểm IoU khi không có mask tay.
- Demo mở và xử lý ảnh mẫu; đã cập nhật `docs/demo_screenshot.png`.
- Báo cáo Word/PDF sinh lại từ kết quả hiện tại; kiểm tra phân trang PDF trước khi đóng gói. Word giữ công thức Office Math; PDF dùng trình dựng ReportLab như hướng dẫn README.

## 5. Các mục chưa thể đánh dấu hoàn thành

| Mục | Trạng thái và việc cần làm |
|---|---|
| IRRI SES theo từng bệnh | Chưa xác minh được bảng chính thức: URL PDF IRRI timeout. Không dùng snippet bị lệch hàng để lập thang. `irri_score=null`, bảng IRRI rỗng; quy ước diện tích được ghi riêng. Cần đọc đúng trang/bệnh/ấn bản rồi mới thêm quy tắc và test ngưỡng. |
| Hướng dẫn BVTV theo tài liệu chuyên môn | Đã có lời khuyên theo dõi/xác nhận, nhưng chưa phải phác đồ BVTV được kiểm chứng. Không ghi lời khuyên tự soạn thành hướng dẫn chính thức. |
| ≥100 ảnh nhóm tự chụp | Hiện 0. Thu ảnh gốc, ID cây/lá/phiên, nhãn có kiểm chứng; khuyến nghị ảnh thẻ xám tham chiếu theo phiên. |
| ≥30 mask tay độc lập | Hiện 0. Cần segval riêng và segtest đã khóa; chuẩn hóa hướng ảnh trước gán nhãn. Sau đó mới tính IoU các phương pháp. |
| Lớp lá khỏe | UCI hiện không có healthy. Cần bổ sung dữ liệu và train/test lại; chưa dùng demo để kết luận lá khỏe. |
| Thực nghiệm cà phê | Khung mã có hỗ trợ crop riêng; chưa có `coffee_knn.npz`. Nguồn BRACOL/RoCoLe và hướng dẫn nhập được ghi trong tài liệu, chưa thay bằng kết quả lúa. |
| Đánh giá ngoài đồng so với nền trơn | Miền field hiện n=0 nên tiếp tục pending. Không gọi stress tổng hợp là thực nghiệm ngoài đồng. |

Không tạo ảnh thực địa, mask tay, điểm IRRI, chữ ký hoặc lịch sử Git để lấp các mục còn thiếu. Code và báo cáo hiện dùng được làm nền thực hiện; đồ án chưa đủ toàn bộ yêu cầu dữ liệu để tuyên bố hoàn thành nộp cuối.

## 6. Cách tiếp tục sau khi có dữ liệu

1. Điền `data/group_map_template.csv` với ID, group theo cây/phiên và split thật.
2. Dùng `scripts/import_folders.py --group-map ... --out data/manifest.csv`; xem `data/README.md` cho lệnh đầy đủ. Script gộp và giữ bản sao manifest.
3. Gán mask tay bằng công cụ đa giác, nhập mask và xác nhận nhãn; không để cây/phiên chung qua các split.
4. Chọn tham số phân đoạn chỉ trên train/segval; khóa cấu hình trước khi đánh giá test/segtest.
5. Chạy kiểm thử, kiểm chứng, thực nghiệm; tạo lại báo cáo và đọc các kết luận theo số liệu mới.
