# Nguồn học thuật và dữ liệu

Các URL được tra cứu ngày 05/11/2026. Bài báo dùng làm nguồn ý tưởng và phương pháp, không phải kết quả thực nghiệm của nhóm. DOI giữ định danh ổn định; truy cập toàn văn có thể phụ thuộc nhà xuất bản.

1. H. B. Prajapati, J. P. Shah, V. K. Dabhi, “Detection and classification of rice plant diseases,” Intelligent Decision Technologies, vol. 11, no. 3, pp. 357–373, 2017. [DOI 10.3233/IDT-170301](https://doi.org/10.3233/IDT-170301).
2. J. Shah, H. Prajapati, V. Dabhi, “Rice Leaf Diseases,” UCI Machine Learning Repository. [DOI 10.24432/C5R013](https://doi.org/10.24432/C5R013). [Trang dữ liệu](https://archive.ics.uci.edu/dataset/486/rice+leaf+diseases).
3. R. A. Krohling, J. G. M. Esgario, J. A. Ventura, “BRACOL,” Mendeley Data, v1, 2019. [DOI 10.17632/yy2k5y8mxg.1](https://data.mendeley.com/datasets/yy2k5y8mxg/1).
4. J. G. M. Esgario, R. A. Krohling, J. A. Ventura, “Deep learning for classification and severity estimation of coffee leaf biotic stress,” Computers and Electronics in Agriculture, vol. 169, 105162, 2020. [DOI 10.1016/j.compag.2019.105162](https://doi.org/10.1016/j.compag.2019.105162). [Bản tác giả](https://arxiv.org/abs/1907.11561).
5. R. M. Haralick, K. Shanmugam, I. Dinstein, “Textural Features for Image Classification,” IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, 1973. [DOI 10.1109/TSMC.1973.4309314](https://doi.org/10.1109/TSMC.1973.4309314).
6. T. Ojala, M. Pietikäinen, T. Mäenpää, “Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns,” IEEE TPAMI, vol. 24, no. 7, pp. 971–987, 2002. [DOI 10.1109/TPAMI.2002.1017623](https://doi.org/10.1109/TPAMI.2002.1017623).
7. S. P. Lloyd, “Least Squares Quantization in PCM,” IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129–137, 1982. [DOI 10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489).
8. S. P. Mohanty, D. P. Hughes, M. Salathé, “Using Deep Learning for Image-Based Plant Disease Detection,” Frontiers in Plant Science, vol. 7, 1419, 2016. [DOI 10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419).
9. R. C. Gonzalez, R. E. Woods, Digital Image Processing, 4th ed., Pearson, 2018. Cơ sở hình thái học, màu sắc, phân đoạn. Đây là giáo trình; số trang phụ thuộc ấn bản.
10. scikit-image developers, [Color API](https://scikit-image.org/docs/stable/api/skimage.color.html) và [Feature API](https://scikit-image.org/docs/stable/api/skimage.feature.html). Đối chiếu HSV, Lab, GLCM.
11. scikit-learn developers, [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Giao thức CV và đối chiếu.
12. OpenCV developers, [Eroding and Dilating](https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html). Quy ước biên được đặt tường minh trong mã kiểm chứng.
13. Streamlit, [st.camera_input](https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input). Widget chụp ảnh.
14. NASA Space Apps, [Field Shift 2026](https://www.spaceappschallenge.org/2026/challenges/field-shift-adapting-farms-with-nasa-data/). Bối cảnh ứng dụng, không phải nguồn nhãn bệnh lá của thực nghiệm.

Không mặc định PlantVillage có lúa/cà phê. Danh sách crop–disease trong bài [8] cần kiểm tra trước khi chọn dữ liệu. Không gắn nhãn NASA cho ảnh UCI/BRACOL.


15. OpenCV, [CLAHE API](https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html). Đối chiếu CLAHE cùng thang clip-limit, lưới ô và quy tắc đệm; dung sai 1 mức xám do làm tròn.
16. SciPy, [distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html); scikit-image, [regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops). Đối chiếu EDT, eccentricity và solidity.
17. IRRI, [Standard Evaluation System for Rice](https://www.knowledgebank.irri.org/images/docs/rice-standard-evaluation-system.pdf). Nguồn sơ cấp cần kiểm tra; truy cập trực tiếp và tải PDF đều timeout trong phiên này. Không chép thang từ snippet có hàng bị lệch, không triển khai điểm IRRI khi chưa xác minh trang bảng/bệnh/ấn bản.
18. J. Parraga-Alava et al., [RoCoLe v2](https://data.mendeley.com/datasets/c5yvn32dzg/2), 2019, DOI 10.17632/c5yvn32dzg.2, CC BY 4.0. Đã kiểm tra trang metadata, chưa dùng trong thực nghiệm hiện tại.

Khuyến nghị trong app là bảng hướng dẫn quan sát và xác nhận chuyên môn do đồ án biên soạn, chưa phải phác đồ BVTV được cơ quan chuyên môn phê duyệt. Không có liều thuốc hoặc ngưỡng can thiệp kinh tế được suy từ ảnh.
