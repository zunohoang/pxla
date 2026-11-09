# Dữ liệu và cách gán nhãn

## Bước 1 · Ảnh lúa công khai

Chạy `python scripts/download_rice.py` tại thư mục gốc. Nguồn UCI Rice Leaf Diseases [DOI 10.24432/C5R013](https://doi.org/10.24432/C5R013), CC BY 4.0, 120 ảnh nền trắng, 40 ảnh/lớp. Script cố định seed 42, tạo 90 ảnh train và 30 ảnh test. Không có lớp lá khỏe, không có nhãn pixel. `rice_provenance.json` lưu URL và SHA256 gói tải. Giữ nguyên ghi công tác giả khi chia sẻ lại.

## Bước 2 · Dữ liệu cà phê

[BRACOL bản 1](https://data.mendeley.com/datasets/yy2k5y8mxg/1), DOI 10.17632/yy2k5y8mxg.1, CC BY 4.0. Nguồn có 1747 ảnh toàn lá và tập triệu chứng cắt riêng. Chỉ dùng **ảnh toàn lá** cho pipeline này; không trộn các crop cùng một lá qua train/test. Nhãn mức độ của nguồn được tạo từ phân đoạn tự động, không thay thế 30 mặt nạ gán tay độc lập.

Tải từ nguồn, đọc metadata, chuẩn hóa nhãn vào `data/raw/coffee/<label>/<image>.jpg`. Gợi ý: healthy, rust, leaf_miner, brown_spot, cercospora. Không tự suy diễn nhãn từ tên file nếu metadata chưa xác nhận. Lớp bệnh hỗn hợp phải có quy tắc bệnh chủ đạo hoặc bài toán đa nhãn riêng.

```powershell
python scripts/import_folders.py data/raw/coffee --crop coffee --domain plain --source https://doi.org/10.17632/yy2k5y8mxg.1 --license "CC BY 4.0" --out data/coffee.csv
```

Script import chỉ khởi tạo CSV. Sửa `group` theo cây/lá/phiên chụp; phân nhóm train/test độc lập, giữ split chính thức nếu có. Gộp các dòng vào `data/manifest.csv`, giữ đúng một dòng header. Cả hai cây sẽ được huấn luyện bởi cùng lệnh `python run_experiments.py`; xuất `models/rice_knn.npz` và `models/coffee_knn.npz`.

## Bước 3 · Tự chụp ít nhất 100 ảnh

Thu ảnh thực tại ruộng/vườn; không lấy ảnh mạng làm ảnh tự chụp. Cố gắng ≥20 cây/địa điểm/phiên độc lập, đa dạng máy và điều kiện sáng. 100 ảnh không đồng nghĩa 100 mẫu độc lập. Ghi nhật ký ngày, cây, lá, phiên, người chụp và người xác nhận bệnh. Không có chuyên gia thì dùng nhãn `chưa xác nhận` trong nhật ký và chưa đưa vào phân lớp.

Trong manifest, `domain=field`, `source=self`, `group=<ma_cay_phien>`; giữ toàn bộ ảnh của cùng nhóm ở một split. Dùng ảnh công khai train, ảnh ngoài đồng test để đo chuyển miền. Nếu dùng ảnh ngoài đồng để tinh chỉnh, tách nhóm field-train và field-test trước. Không tăng cường dữ liệu trước khi chia nhóm. Không tạo lịch sử thu thập hoặc chữ ký thay thành viên.

## Bước 4 · Tạo nhãn pixel trên ít nhất 30 ảnh

Mở `docs/annotate.html` bằng trình duyệt. Vẽ đa giác toàn phần lá nhìn thấy và từng vết tổn thương, xuất JSON. Có thể dùng LabelMe với nhãn `leaf`, `lesion` và shape_type polygon. Ảnh có EXIF xoay cần lưu lại hướng hiển thị chuẩn trước khi gán nhãn. Lưu ảnh và mask cùng kích thước gốc, mask xám 0/255, nền 0. Không vẽ mặt nạ từ kết quả dự đoán của thuật toán.

```powershell
python scripts/convert_annotations.py data/annotations/anh01.json --out data/masks
```

Điền `leaf_mask`, `lesion_mask` vào manifest. Dành **30 ảnh độc lập cho segtest** để báo IoU; ảnh điều chỉnh ngưỡng nằm ở segval riêng, không dùng lại trong segtest. Nếu chỉ có 30 ảnh tổng cộng thì dùng giao thức CV phân đoạn hoặc thu thêm, không chỉnh ngưỡng trên toàn 30 ảnh rồi gọi đó là test. Ít nhất hai người gán nhãn một phần ảnh; trao đổi và thống nhất vùng không chắc chắn trước khi chấm. Lưu mặt nạ đã thống nhất và người duyệt.

## Bước 5 · Kiểm tra manifest

`id` duy nhất; `path` tính từ thư mục gốc; `crop` là rice/coffee; `label` là nhãn đã kiểm tra; `domain` plain/field; `group` định danh lần thu thập; `split` train/test/segval/segtest. Kiểm tra ảnh trùng byte bằng SHA256 đã tự động hóa. Ảnh gần trùng phải rà soát thủ công; đổi tên hay nén JPEG không làm chúng thành mẫu độc lập.

Phiên bản PlantVillage phổ biến không có lúa/cà phê. Chỉ chọn dữ liệu khi danh sách loài và bệnh phù hợp. Trang NASA Field Shift cung cấp bối cảnh và dữ liệu quan sát Trái Đất; không coi ảnh vệ tinh là nhãn bệnh pixel của lá chụp điện thoại.


## Cập nhật v2

Script tải UCI giữ các dòng đã có và chỉ thêm ID còn thiếu; CSV được sao lưu trước khi ghi. Script import cũng gộp an toàn, không ghi đè group/split/mask thủ công. Đối với ảnh field, chuẩn bị `data/group_map_template.csv` với id,group,split thật rồi chạy:

```powershell
python scripts/import_folders.py data/field/rice --crop rice --domain field --source self --license own --group-map data/field_groups.csv --out data/manifest.csv
```

Có thể đặt thẻ xám ở mép khung trong một ảnh tham chiếu mỗi phiên để kiểm tra chiếu sáng; ảnh phân tích chính vẫn cần một lá rõ, không bị vật tham chiếu che. Ghi ID lá, cây, máy và phiên; không tự sinh ID cây để thay bằng chứng thu thập. Thu thêm lá khỏe có người xác nhận. Mask lá gán tay phải không rỗng; mask tổn thương được rỗng với lá không có vết.

Nguồn cà phê bổ sung: [RoCoLe v2](https://data.mendeley.com/datasets/c5yvn32dzg/2), 1560 ảnh robusta, CC BY 4.0, DOI 10.17632/c5yvn32dzg.2. Nhãn nguồn gồm khỏe/bệnh và mức độ; không tự chuyển nhãn bệnh tổng quát thành tác nhân cụ thể nếu metadata không xác nhận. Pipeline chưa tải hoặc huấn luyện trên RoCoLe/BRACOL trong lần chạy này.
