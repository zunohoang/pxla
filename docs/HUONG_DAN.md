# Lộ trình thực hiện và hoàn thiện bài tập lớn

## Những quy định đã được đối chiếu

File giảng viên yêu cầu thuật toán cốt lõi tự cài NumPy, kiểm chứng thư viện, demo có GUI, báo cáo 5 chương, tài liệu IEEE ≥10 nguồn và ≥3 bài báo có DOI. Nội dung chính 25–40 trang, sản phẩm PDF 30–60 trang; Chương 3 và 4 chiếm hơn nửa nội dung. A4, Times New Roman 13 pt, giãn dòng 1,3; lề trái 3 cm, phải 2 cm, trên/dưới 2,5 cm. Cần video 5–7 phút và slide 15–20 trang khi nộp.

File có hai điểm chưa thống nhất: thời gian 3 tuần nhưng phần Git nhắc 8 tuần; phần phụ lục vừa gợi ý chỉ để GitHub vừa nêu các phụ lục A/B/C. Đề 13 trong file là Hough làn đường; đề bệnh lá do bạn cung cấp là phạm vi của dự án này. Nên xác nhận lịch và số đề với giảng viên; tiếp tục dùng nội dung kỹ thuật hiện tại. Không tạo commit lùi ngày hoặc chữ ký giả để bù quy định.

## Bước 1 · Chốt phạm vi và đọc kết quả có sẵn

Xem README, mở `results/metrics.json` nếu đã chạy, hiểu số ảnh và lớp của từng cây. Tên đề tài nên là “Phân tích bệnh lá lúa và cà phê từ ảnh điện thoại bằng xử lý ảnh truyền thống”. Từ “chẩn đoán” được hiểu là hỗ trợ sàng lọc trong đồ án; không suy ra tác nhân bệnh chỉ từ màu vết.

Mục tiêu kiểm chứng: 5 chức năng chạy được, thuật toán lõi đối chiếu đạt dung sai, có macro-F1 và accuracy trên test tách nhóm, có IoU trên ≥30 ảnh gán tay, có so sánh miền nền trơn/ngoài đồng. Mục tiêu accuracy ≥0,80 hoặc IoU ≥0,50 nếu đặt là **mục tiêu kỳ vọng**, không phải kết quả. Báo cáo không được sửa mục tiêu sau khi xem test để tuyên bố đạt.

## Bước 2 · Cài môi trường và chạy thử

Dùng lệnh trong README. Nếu PowerShell chặn activate, gọi trực tiếp `.venv\Scripts\python.exe`; không cần đổi ExecutionPolicy. Không có GPU vẫn chạy được. Cài lock để dùng đúng bản đã thử. Khi gặp lỗi DLL của OpenCV, kiểm tra Python 64 bit và không cài đồng thời nhiều gói OpenCV. Không tự thay seed giữa các lần chạy để chọn kết quả đẹp.

## Bước 3 · Hiểu thuật toán theo luồng

1. Ảnh RGB có hướng EXIF được chuẩn hóa, cạnh dài tối đa 256 px, giữ tỉ lệ khung.
2. ExG chọn vùng xanh; HSV mở rộng sang vàng/nâu. Mở/đóng giảm nhiễu; chọn thành phần nối với nhiều seed xanh nhất; lấp lỗ bảo toàn vết bệnh trong lá.
3. Chuyển Lab; K-means chỉ nhận pixel trong lá. Cụm được chọn theo tọa độ tâm màu, không theo số thứ tự cụm.
4. Tính `lesion.sum()/leaf.sum()`. Tử số phải là tập con mẫu số.
5. Trích 74 đặc trưng; chuẩn hóa bằng trung bình/độ lệch chuẩn tập train; phân lớp k-NN. Không dùng đặc trưng test khi fit scaler.
6. Nếu khoảng cách lớn hoặc thiếu đồng thuận, demo trả “Chưa đủ bằng chứng”. Tỉ lệ phiếu không được gọi là xác suất bệnh.

## Bước 4 · Thu thập và gán nhãn

Thực hiện `data/README.md`. Đặt tên cây và phiên chụp trước khi thu ảnh. Một người chụp, một người ghi biểu mẫu; lưu nhãn chưa chắc chắn riêng. Với cà phê, lá có thể có nhiều tác nhân cùng lúc; chỉ nhập nhãn đơn nếu có quy tắc bệnh chủ đạo rõ ràng. Bổ sung lá khỏe để tránh mô hình buộc mọi ảnh vào một trong các bệnh.

Đừng chọn riêng ảnh đẹp để đánh giá. Lấy mẫu theo miền, lớp và điều kiện sáng; bao gồm ảnh khó. 30 mask segtest không dùng để chỉnh quy tắc màu. Ghi nguyên nhân loại ảnh trước khi chạy; báo tỷ lệ thất bại trên toàn bộ ảnh được nhận.

## Bước 5 · Chạy ba nhóm thí nghiệm

**So sánh phương pháp:** k-NN NumPy và SVM RBF thư viện trên cùng train/test; CV chỉ nằm trong train. Báo cáo danh sách lớp, confusion matrix, accuracy, macro-F1. Không so sánh điểm tuyệt đối với bài báo dùng dữ liệu khác để xếp hạng.

**Khảo sát tham số:** k=1,3,5,7 và 3 tập đặc trưng màu/kết cấu/toàn bộ, lấy trung bình ± độ lệch chuẩn CV. Macro-F1 CV dùng chọn cấu hình; test chỉ dùng đánh giá cuối. K=3 phân đoạn là mặc định cố định; nếu khảo sát K=2,3,4,5, phải chọn bằng segval, không bằng segtest.

**Điều kiện bất lợi:** test nền trơn, test ngoài đồng; ảnh test nhân độ sáng 0,6 và 1,4 chỉ là kiểm tra stress tổng hợp. Không ghi các biến thể này là ảnh thực địa. Giữ số thất bại tách lá trong báo cáo; accuracy trên ảnh thành công phải ghi rõ mẫu số.

## Bước 6 · Đọc và viết phân tích

Nêu con số cùng cỡ mẫu, rồi giải thích bằng đặc trưng/điều kiện chụp. Sai giữa bạc lá và đốm nâu có thể liên quan vùng vàng/nâu gần nhau; đây là giả thuyết, cần xem ảnh sai để kiểm tra. IoU thấp nhưng accuracy cao có thể do bộ phân lớp nhận diện màu tổng thể, không chứng minh mặt nạ đúng. Accuracy cao trên 30 ảnh có khoảng bất định lớn và có thể chịu ảnh hưởng cách chia ảnh.

Tách kết luận về đúng thuật toán, đúng trên dữ liệu công khai và khả năng áp dụng ngoài đồng. Đối chiếu thư viện đạt không có nghĩa mô hình chẩn đoán đáng tin cậy. Không báo “giảm thuốc” hoặc “phát hiện sớm” như hiệu quả đã được đo nếu không có thử nghiệm can thiệp.

## Bước 7 · Hoàn thiện hồ sơ nhóm

Điền trường/lớp/giảng viên/tên 5 thành viên trên bìa. Bảng vai trò là đề xuất, không phải đóng góp đã xác nhận. Mỗi thành viên commit phần mình đã hiểu và tự kiểm tra; ghi họp thật vào `docs/nhat_ky_nhom.csv`; điền đánh giá chéo vào `docs/danh_gia_cheo.csv`; ký sau khi thống nhất. Thay liên kết GitHub chờ điền trong báo cáo bằng repo thật.

Khai báo AI: dùng Codex hỗ trợ cấu trúc, code và văn bản; nhóm kiểm chứng công thức bằng ví dụ tay, test, thư viện và thực nghiệm thực tế. Khi chưa thực hiện các bước nhóm tự kiểm tra, không ký cam đoan rằng đã làm.

## Bước 8 · Demo và bảo vệ

Luyện bài nói 12 phút, demo 3 phút, vấn đáp 5 phút. Mở demo trước, tắt tải mạng để kiểm tra pipeline hoạt động offline khi model đã lưu. Chuẩn bị một ảnh dễ, một ảnh ngoài đồng khó, một ảnh không phù hợp để minh họa từ chối và giới hạn. Quay video 5–7 phút với thao tác thật của nhóm; không sử dụng video mô phỏng thay thao tác thực tế.

## Phân công đề xuất trong ba tuần

| Người | Mã và việc phụ trách | Sản phẩm kiểm tra |
|---|---|---|
| TV1 | app.py, tích hợp, README | Demo chạy từ máy sạch, xử lý lỗi |
| TV2 | lý thuyết, verify_algorithms.py | Ví dụ tay, bảng max error/MSE, tài liệu nguồn |
| TV3 | algorithms.py, leaf_mask | Đối chiếu HSV/Lab/hình thái/K-means |
| TV4 | features, model.py | LBP/GLCM/k-NN, cơ chế từ chối, SVM đối chiếu |
| TV5 | manifest, masks, run_experiments.py | Nguồn dữ liệu, chia nhóm, bảng/hình thực nghiệm |

Tuần 1 chốt nhãn, đọc lý thuyết, chạy baseline và thu ảnh. Tuần 2 hoàn tất gán nhãn, kiểm chứng và điều chỉnh bằng train/segval. Tuần 3 khóa cấu hình, đánh giá test, viết phân tích, quay video và luyện vấn đáp. Nếu lịch môn là 8 tuần, phân tán công việc theo lịch thật, không tạo nhật ký hồi tố.

## Điều kiện trước khi nộp

- Các kiểm thử chạy đạt; đúng commit/manifest/model tương ứng.
- Cà phê đã được nhập và có kết quả nếu giữ cả hai cây trong phạm vi thực nghiệm.
- ≥100 ảnh tự chụp có nhật ký; ≥30 mask test gán tay có người duyệt.
- Có accuracy, macro-F1, IoU, confusion matrix và so sánh miền thực tế.
- Báo cáo PDF đã cập nhật số liệu mới; slide 15–20 trang và video thật 5–7 phút.
- Điền danh tính, đóng góp, chữ ký, liên kết repo; không có số liệu giả hoặc kết quả mẫu bị gọi là thực địa.
