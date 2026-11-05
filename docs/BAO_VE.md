# Kịch bản bảo vệ

## Nội dung 18 slide trong 12 phút

1. Tên đề tài, thành viên và đầu ra demo (20 giây).
2. Tình huống tại ruộng/vườn Việt Nam và vai trò hỗ trợ sàng lọc (30 giây).
3. Đầu vào, đầu ra, 5 chức năng bắt buộc (30 giây).
4. Nguồn dữ liệu, giấy phép, các lớp thật đang có (40 giây).
5. Chia theo cây/phiên, train/CV/test và tránh rò rỉ (40 giây).
6. Sơ đồ pipeline tổng thể (30 giây).
7. ExG/HSV và cách giữ phần lá bị vàng/nâu (40 giây).
8. Mở/đóng, thành phần liên thông, lấp lỗ (35 giây).
9. Lab và K-means; không gán ý nghĩa bệnh cho cluster ID (45 giây).
10. Tỉ lệ diện tích, ví dụ 120/1000=12%, quy ước mức độ (35 giây).
11. Histogram, Lab, LBP vuông và GLCM có mask (45 giây).
12. Chuẩn hóa train-only, k-NN, SVM đối chiếu, từ chối (40 giây).
13. Bảng kiểm chứng max error/MSE của thuật toán tự viết (35 giây).
14. So sánh k-NN/SVM và ma trận nhầm lẫn từ lần chạy thật (55 giây).
15. Khảo sát k và tập đặc trưng bằng CV (45 giây).
16. IoU/ngoài đồng và kiểm tra độ sáng; đánh dấu phần chưa hoàn thành (50 giây).
17. Hạn chế: nền xanh, ảnh gần trùng, không có lá khỏe, severity chưa hiệu chuẩn (40 giây).
18. Kết luận đối chiếu mục tiêu và hướng MobileNet/GPS/vệ tinh (30 giây).

Tổng khoảng 12 phút. Cập nhật slide 4, 14, 15, 16 sau lần chạy cuối. Nếu chưa có dữ liệu thực địa, trình bày kế hoạch và trạng thái còn thiếu, không đọc kết quả UCI thành kết quả ngoài đồng.

## Demo 3 phút    

0:00–0:30 mở giao diện, chọn cây, nêu danh sách lớp model. 0:30–1:20 tải ảnh hoặc chọn mẫu; chỉ ảnh gốc, mask lá, overlay vết nghi bệnh. 1:20–1:50 giải thích tỉ lệ, nhãn bệnh và mức độ. 1:50–2:20 xem ảnh khó, nêu khả năng từ chối và giới hạn. 2:20–3:00 xuất PNG/JSON, nhắc dữ liệu/phiên bản và khuyến nghị xác nhận chuyên môn.

## Video demo 5 đến 7 phút

0:00–0:40 giới thiệu vấn đề và nhóm; 0:40–1:30 cài/chạy từ README; 1:30–3:40 thao tác thật và giải thích từng bước; 3:40–4:40 ảnh khó và sai số; 4:40–5:40 bảng thực nghiệm; 5:40–6:10 giới hạn và kết luận. Dùng quay màn hình thật, không bịa thao tác hay kết quả.

## Câu hỏi cần tự trả lời

**Vì sao dùng Lab?** Khoảng cách Euclid trên màu Lab thường phù hợp với việc gom màu hơn RGB mã hóa sRGB; không bảo đảm phân biệt triệu chứng với thay đổi ánh sáng.

**K-means có tự biết bệnh không?** Không. Nó chỉ giảm SSE, quy tắc chọn cụm màu tạo ứng viên tổn thương. Nhãn tác nhân bệnh do bộ phân lớp học từ dữ liệu đã xác nhận.

**K-means hội tụ tại tối ưu toàn cục không?** Không. Gán cụm và cập nhật trung bình đều không làm tăng mục tiêu, nhưng có thể dừng ở nghiệm cục bộ. Seed và khởi tạo ảnh hưởng kết quả.

**Tại sao không chỉ ngưỡng xanh?** Ngưỡng xanh có thể loại bỏ mô lá bị vàng/nâu, làm sai cả tử và mẫu của tỉ lệ diện tích. HSV mở rộng và lấp lỗ giảm một phần lỗi nhưng không khôi phục mô bị mất ở rìa.

**LBP này có giống skimage không?** Không hoàn toàn. Dự án dùng 8 điểm vuông 3×3, còn LBP tròn dùng nội suy ở điểm chéo; vì vậy kiểm chứng LBP bằng định nghĩa/ví dụ tay và đối chiếu các phần có cùng quy ước.

**Vì sao phải mask GLCM cả hai đầu pixel?** Nếu chỉ mask điểm đầu, cặp kết cấu có thể chứa nền và làm đặc trưng phụ thuộc nền.

**Vì sao không fit scaler trước CV?** Vì trung bình/độ lệch chuẩn lúc đó chứa thông tin validation, làm đánh giá lạc quan.

**Accuracy khác macro-F1 ở đâu?** Accuracy đếm đúng toàn mẫu. Macro-F1 lấy trung bình F1 từng lớp với trọng số ngang nhau, giúp nhìn rõ lớp ít ảnh.

**IoU hai mask rỗng bằng bao nhiêu?** Mã quy ước 1; cần báo thêm số ảnh GT rỗng và IoU trên ảnh có bệnh để tránh điểm cao do nhiều ảnh khỏe.

**Tại sao số liệu công bố không so trực tiếp?** Bộ dữ liệu, split, nhãn và độ khó khác nhau. Chỉ xem công trình trước là bối cảnh; xếp hạng trực tiếp cần cùng giao thức.

**Có thể giảm phun thuốc không?** Đó là động lực ứng dụng. Dự án chưa đo lượng thuốc hay hiệu quả can thiệp nên không tuyên bố đã giảm.

**Vì sao cần ảnh ngoài đồng?** Mô hình có thể dựa vào màu/nền chuẩn hóa của bộ công khai. Ảnh chụp tại Việt Nam có khác biệt máy, giống, ánh sáng và môi trường; đây là chuyển miền cần đo riêng.
