# LeafLab: giải thích dự án từ A đến Z

Tài liệu này đi theo đúng thứ tự dữ liệu chảy qua code. Ở mỗi bước có bốn phần: **tên thuật toán**, **cách hoạt động**, **lý do dùng**, **công thức**. Mọi tham số đều lấy từ code hiện tại (schema `leaflab-3`).

---

## Mục lục

1. [Bài toán và ý tưởng tổng thể](#1-bài-toán-và-ý-tưởng-tổng-thể)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Hai luồng chạy: demo và huấn luyện](#3-hai-luồng-chạy-demo-và-huấn-luyện)
4. [Bước 1 – Đọc ảnh](#4-bước-1--đọc-ảnh)
5. [Không gian màu: RGB, HSV, CIE L\*a\*b\*](#5-không-gian-màu-rgb-hsv-cie-lab)
6. [Bước 2 – Tiền xử lý: mặt nạ chói, Gray-world, CLAHE, Median](#6-bước-2--tiền-xử-lý)
7. [Bước 3 – Tách lá: ExG, HSV, Otsu, ExR, hình thái học, liên thông, lấp lỗ](#7-bước-3--tách-lá)
8. [Bước 4 – Tìm vết bệnh: K-means++ trên a\*b\*](#8-bước-4--tìm-vết-bệnh)
9. [Bước 5 – Tỷ lệ diện tích, mức độ, cảnh báo](#9-bước-5--tỷ-lệ-diện-tích-mức-độ-cảnh-báo)
10. [Bước 6 – Trích 403 đặc trưng: histogram, moment, GLCM, LBP, hình dạng, gradient](#10-bước-6--trích-403-đặc-trưng)
11. [Bước 7 – Phân lớp: chuẩn hóa, k-NN, LDA, Kernel Ridge, cơ chế từ chối](#11-bước-7--phân-lớp)
12. [Bước 8 – Khuyến nghị](#12-bước-8--khuyến-nghị)
13. [Huấn luyện và đánh giá: CV phân nhóm, macro-F1, IoU/Dice, stress test](#13-huấn-luyện-và-đánh-giá)
14. [Lưu và nạp mô hình an toàn](#14-lưu-và-nạp-mô-hình)
15. [Kết quả hiện tại và giới hạn](#15-kết-quả-hiện-tại-và-giới-hạn)
16. [Bảng tra cứu tham số](#16-bảng-tra-cứu-tham-số)

---

## 1. Bài toán và ý tưởng tổng thể

**Đầu vào:** một ảnh chụp **một lá** lúa hoặc cà phê.

**Đầu ra:**
1. **Loại bệnh** dự đoán. Hệ thống có thể từ chối trả lời: "Chưa đủ bằng chứng".
2. **Tỷ lệ diện tích nghi tổn thương** trên lá (%).
3. **Mức độ** theo quy ước đồ án: chưa thấy, nhẹ, trung bình, nặng.
4. **Khuyến nghị** thận trọng, không kê thuốc.

**Ràng buộc thiết kế:** chỉ dùng **xử lý ảnh cổ điển** và **học máy cổ điển**, tự cài bằng NumPy. Không dùng CNN hay mạng nơ-ron tách nền. OpenCV GrabCut và sklearn SVM chỉ là **phương án đối chứng** để so sánh.

**Ý tưởng chính:**

```
Ảnh → chuẩn hóa ánh sáng → tách lá khỏi nền → tìm vùng màu bất thường trên lá (vết bệnh)
    → đo % diện tích → trích đặc trưng màu/kết cấu/hình dạng → bộ phân lớp → bệnh gì
```

Lúa và cà phê dùng **cùng pipeline và cùng tham số xử lý ảnh**. Chỉ khác ở **mô hình phân lớp** (mỗi cây một file) và **danh sách lớp bệnh**:

| Cây | Nguồn dữ liệu | Lớp | Train / Test |
|---|---|---|---|
| Lúa | UCI Rice Leaf Diseases | `bacterial_leaf_blight` (bạc lá), `brown_spot` (đốm nâu), `leaf_smut` (đốm than) | 90 / 30 |
| Cà phê | BRACOL | `healthy`, `rust` (gỉ sắt), `leaf_miner` (sâu vẽ bùa), `cercospora` (đốm mắt cua), `brown_spot` | 1007 / 335 |

Dữ liệu lúa **không có lớp lá khỏe**, nên app cảnh báo khi dùng mô hình lúa để kết luận "khỏe hay bệnh".

---

## 2. Cấu trúc thư mục

```
app.py                  Demo Streamlit (tải ảnh / camera / ảnh mẫu)
run_experiments.py      Trích đặc trưng, CV chọn mô hình, test, stress, lưu models/ và results/
START_DEMO.bat          Chạy nhanh demo trên Windows

src/
  algorithms.py         Toàn bộ thuật toán NumPy: màu, CLAHE, Otsu, hình thái, liên thông,
                        K-means, LBP, GLCM, distance transform, bao lồi, LDA, RBF, ridge
  pipeline.py           Config + load_rgb → preprocess → leaf_mask → lesion_mask → process
  features.py           Lược đồ 403 đặc trưng (FEATURE_NAMES / FEATURE_GROUPS / FEATURE_SCHEMA)
  model.py              KNN (tùy chọn LDA), KernelRidge, lưu/nạp .npz, giải thích
  advice.py             severity() và recommendation()
  data.py               Đọc/kiểm tra/ghép manifest.csv
  evaluation.py         accuracy, macro-F1, confusion; IoU, Dice

data/
  manifest.csv          Mỗi ảnh một dòng: id, path, crop, label, domain, group, split, mask...
  severity_rules.json   Ngưỡng mức độ và câu khuyến nghị
  raw/                  Ảnh gốc (không đưa lên git)
  samples/              24 ảnh test để demo khi không có raw/

models/                 rice_knn.npz, coffee_knn.npz (tên giữ "knn", trường method cho biết loại thật)
results/                metrics.json, cross_validation.json, predictions.csv, *_confusion.png ...
scripts/                Tải dữ liệu, kiểm chứng thuật toán, so sánh tách lá, GrabCut đối chứng, báo cáo
tests/                  pytest
```

---

## 3. Hai luồng chạy: demo và huấn luyện

### 3.1 Luồng demo – `streamlit run app.py`

```
main()
 ├─ chọn crop ∈ {rice, coffee}
 ├─ KNN.load("models/<crop>_knn.npz")  → (model, config, crop)     [model.py]
 │     kiểm tra schema, số chiều, tên đặc trưng; trả về KNN hoặc KernelRidge
 ├─ lấy ảnh (upload / camera / ảnh mẫu)
 ├─ load_rgb(source, config.max_side)                              [pipeline.py]
 ├─ process(rgb, config)                                           [pipeline.py]
 │     ├─ preprocess()     → processed, glare
 │     ├─ leaf_mask(rgb)   → leaf
 │     ├─ lesion_mask(processed, leaf, valid=~glare) → lesion, centers
 │     ├─ ratio, severity, warnings, overlay
 │     └─ features(processed, leaf, lesion, valid, raw=rgb) → vector 403 chiều
 ├─ model.predict_details(features)  → label, vote, distance, rejected
 ├─ recommendation(label, ratio, rejected)                         [advice.py]
 └─ hiển thị 5 tab + model.explain() + nút tải JSON
```

Điểm quan trọng: `config` được **nạp từ trong file mô hình**, nên lúc demo luôn dùng đúng tham số lúc huấn luyện.

### 3.2 Luồng huấn luyện – `python run_experiments.py`

```
read_manifest()  → kiểm tra dữ liệu, chống rò rỉ giữa train/test
for crop in {coffee, rice}:
    for ảnh in train ∪ test:  load_rgb → process → vector 403      (ma trận X: N×403)
    StratifiedGroupKFold (≤5 fold) chỉ trên train
    for cấu hình in lưới (kNN | LDA+kNN | KernelRidge) × nhóm đặc trưng:
        macro-F1 trung bình qua các fold
    chọn cấu hình tốt nhất → fit lại trên toàn bộ train → save models/<crop>_knn.npz
    chấm test: accuracy, macro-F1, coverage, accepted_accuracy, confusion
    đối chứng SVM; stress (sáng/tối, mờ, xoay); ablation tiền xử lý
đánh giá mask (IoU/Dice) nếu có mask tay ở split segtest
ghi results/*.json, *.csv, *.png
```

---

## 4. Bước 1 – Đọc ảnh

**Hàm:** `load_rgb(source, max_side=256)` trong `pipeline.py`.

**Cách hoạt động:**
1. `ImageOps.exif_transpose`: xoay ảnh theo thẻ EXIF. Ảnh điện thoại thường lưu nằm ngang kèm thẻ "cần xoay".
2. Chuyển sang RGB 8 bit.
3. Ảnh có chiều nào nhỏ hơn 8 px thì báo lỗi.
4. `thumbnail((256, 256), LANCZOS)`: thu nhỏ để **cạnh dài ≤ 256**, giữ tỷ lệ khung.

**Tại sao:**
- Giảm kích thước giúp mọi bước sau (K-means, liên thông, GLCM) chạy nhanh, khoảng vài trăm ms mỗi ảnh. Kích thước cố định còn giúp các đặc trưng so sánh được giữa các ảnh.
- **Lanczos** là phép nội suy dùng hàm sinc có cửa sổ. Khi thu nhỏ, nó ít bị răng cưa (aliasing) hơn nội suy song tuyến tính:

$$L(x)=\begin{cases}\operatorname{sinc}(x)\operatorname{sinc}(x/a) & |x|<a\\ 0 & \text{ngược lại}\end{cases},\qquad \operatorname{sinc}(x)=\frac{\sin \pi x}{\pi x},\ a=3$$

---

## 5. Không gian màu: RGB, HSV, CIE L\*a\*b\*

Pipeline đổi màu nhiều lần. Mỗi không gian màu phục vụ một mục đích riêng.

### 5.1 RGB và sắc độ chuẩn hóa

Pixel gồm $(R,G,B)\in[0,255]^3$. **Sắc độ chuẩn hóa** (chromaticity) bỏ đi độ sáng:

$$r=\frac{R}{R+G+B},\quad g=\frac{G}{R+G+B},\quad b=\frac{B}{R+G+B},\qquad r+g+b=1$$

Cùng một lá dưới nắng `(60,140,50)` và trong bóng râm `(30,70,25)` đều cho $(r,g,b)=(0.24,0.56,0.20)$.

### 5.2 HSV – `rgb_to_hsv`

Đặt $x=\text{RGB}/255$, $M=\max(r,g,b)$, $m=\min(r,g,b)$, $\Delta=M-m$:

$$V=M,\qquad S=\begin{cases}\Delta/M & M>0\\0\end{cases}$$

$$H=60^\circ\times\begin{cases}\left(\frac{G-B}{\Delta}\right)\bmod 6 & M=R\\[2pt] \frac{B-R}{\Delta}+2 & M=G\\[2pt] \frac{R-G}{\Delta}+4 & M=B\end{cases}\qquad(H=0 \text{ khi } \Delta=0)$$

- **H (Hue, sắc màu)**: góc trên vòng màu. 0° đỏ, 60° vàng, 120° xanh lá, 180° xanh ngọc, 240° xanh dương, 300° tím.
- **S (Saturation, độ bão hòa)**: 0 là xám/trắng, 1 là màu thuần.
- **V (Value, độ sáng)**.

**Tại sao dùng HSV:** cho phép viết luật "màu gì" (theo H) tách riêng với "đậm hay nhạt" (theo S) và "sáng hay tối" (theo V). Dùng ở bước tách lá và mặt nạ chói.

### 5.3 CIE L\*a\*b\* – `rgb_to_lab`, `lab_to_rgb`

Chuỗi biến đổi theo chuẩn sRGB, điểm trắng D65:

1. **Bỏ gamma** (tuyến tính hóa), với $c\in[0,1]$:
$$c_{lin}=\begin{cases}c/12.92 & c\le0.04045\\ \left(\frac{c+0.055}{1.055}\right)^{2.4} & c>0.04045\end{cases}$$
2. **Sang XYZ**:
$$\begin{bmatrix}X\\Y\\Z\end{bmatrix}=\begin{bmatrix}0.4124564&0.3575761&0.1804375\\0.2126729&0.7151522&0.0721750\\0.0193339&0.1191920&0.9503041\end{bmatrix}\begin{bmatrix}R_{lin}\\G_{lin}\\B_{lin}\end{bmatrix}$$
   rồi chia cho điểm trắng $(X_n,Y_n,Z_n)=(0.95047,\,1,\,1.08883)$.
3. **Sang Lab**, với $\delta=6/29$:
$$f(t)=\begin{cases}\sqrt[3]{t} & t>\delta^3\\ \frac{t}{3\delta^2}+\frac{4}{29}&\text{ngược lại}\end{cases}$$
$$L^*=116f(Y)-16,\quad a^*=500\,[f(X)-f(Y)],\quad b^*=200\,[f(Y)-f(Z)]$$

Ý nghĩa ba kênh:
- **L\***: độ sáng, từ 0 đến 100.
- **a\***: âm là xanh lá, dương là đỏ/hồng.
- **b\***: âm là xanh dương, dương là vàng.

**Tại sao dùng Lab:**
- Lab được thiết kế **gần đều về cảm nhận**: khoảng cách Euclid trong Lab xấp xỉ mức khác biệt màu mà mắt người thấy, gọi là $\Delta E$. Nhờ vậy K-means dùng khoảng cách Euclid trên Lab là có ý nghĩa.
- **Độ sáng (L\*) tách hẳn khỏi màu (a\*, b\*)**:
  - CLAHE chỉ chỉnh L\* nên tăng tương phản mà không đổi màu.
  - Tìm vết bệnh chỉ dùng a\*b\* nên bóng râm hay chói nhẹ ít ảnh hưởng.
- Lá khỏe có a\* âm. Vết bệnh vàng/nâu/cam có a\* tăng và b\* dương, nên tách rõ trên mặt phẳng a\*b\*.

---

## 6. Bước 2 – Tiền xử lý

**Hàm:** `preprocess(rgb, config)`. Trả về `processed` (ảnh đã chuẩn hóa) và `glare` (mặt nạ chói).

### 6.1 Mặt nạ chói (glare)

$$\text{glare}=(V\ge0.98)\wedge(S\le0.10)$$

**Tại sao:** chỗ phản quang trắng xóa đã mất thông tin màu. Nếu để lại, K-means có thể coi nó là "vết bệnh" vì nó khác màu lá. Mặt nạ được tính trên **ảnh gốc** (trước khi chỉnh) để không bị CLAHE làm lệch. Pixel chói nằm trong lá bị loại khỏi phân cụm và khỏi đặc trưng tổn thương. Nếu chói chiếm hơn 3% lá thì app hiện cảnh báo.

### 6.2 Cân bằng trắng Gray-world – `gray_world`

**Giả định:** trung bình màu của cả cảnh phải là xám trung tính. Nếu ảnh bị ám vàng (đèn sợi đốt) hay ám xanh (bóng cây), kênh nào trung bình lệch thì nhân bù lại.

$$\bar c=\text{mean}(\text{kênh }c),\qquad \text{gain}_c=\operatorname{clip}\!\left(\frac{(\bar R+\bar G+\bar B)/3}{\bar c},\ 0.5,\ 2\right),\qquad c'=\operatorname{clip}(\operatorname{round}(c\cdot\text{gain}_c),0,255)$$

**Tại sao giới hạn gain trong [0.5, 2]:** ảnh lá có rất nhiều màu xanh, nên giả định "trung bình là xám" bị vi phạm nặng. Không giới hạn thì thuật toán sẽ khử xanh quá mạnh và lá thành xám. Giới hạn gain giữ lại hiệu chỉnh vừa phải.

### 6.3 CLAHE trên L\* – `clahe`

**Tên đầy đủ:** Contrast Limited Adaptive Histogram Equalization, cân bằng histogram thích nghi có giới hạn tương phản.

**Nền tảng: cân bằng histogram (HE).** Với ảnh xám 8 bit có histogram $h(v)$ và tổng $N$ pixel, ánh xạ

$$T(v)=\operatorname{round}\!\left(255\cdot\frac{\sum_{u\le v}h(u)}{N}\right)$$

làm các mức xám trải đều trên [0, 255], nên tương phản tăng.

**Hai cải tiến của CLAHE:**
1. **Thích nghi (Adaptive):** chia ảnh thành lưới $8\times8$ ô (`clahe_tiles=8`). Mỗi ô có histogram và LUT riêng, nên vùng tối và vùng sáng được chỉnh riêng.
2. **Giới hạn (Contrast Limited):** cắt mỗi bin histogram ở mức
$$\text{limit}=\max\!\left(1,\ \left\lfloor \text{clip}\cdot\frac{\text{diện tích ô}}{256}\right\rfloor\right),\quad \text{clip}=2$$
   Phần dư được **chia đều lại** cho 256 bin. Phần lẻ còn lại rải theo bước đều. Việc cắt này ngăn vùng phẳng (nền, lá đồng màu) bị khuếch đại nhiễu.
3. **Nội suy song tuyến tính** giữa LUT của 4 ô lân cận để không lộ đường ranh giới ô:
$$T(v)=(1-\alpha_y)\big[(1-\alpha_x)T_{00}(v)+\alpha_xT_{01}(v)\big]+\alpha_y\big[(1-\alpha_x)T_{10}(v)+\alpha_xT_{11}(v)\big]$$

**Cách áp dụng trong project:**
1. RGB → Lab.
2. Đổi L\* (0–100) sang 0–255.
3. Chạy CLAHE.
4. Đổi về 0–100.
5. Lab → RGB.

Bản cài này khớp với quy ước của OpenCV (đệm phản chiếu, nội suy tại tâm ô).

**Tại sao chỉ chỉnh L\*:** tăng chi tiết sáng tối (đốm nhỏ, gân lá) mà **không làm sai màu**. Chỉnh trên từng kênh RGB sẽ làm lệch hue.

### 6.4 Lọc trung vị – `median_filter` (mặc định **tắt**)

Mỗi pixel được thay bằng **trung vị** của cửa sổ $3\times3$, tính riêng từng kênh:

$$I'(y,x)=\operatorname{median}\{I(y+i,x+j)\mid i,j\in\{-1,0,1\}\}$$

Bộ lọc này khử nhiễu muối tiêu mà giữ biên tốt hơn lọc trung bình. Mặc định tắt vì nó cũng xóa các chấm bệnh rất nhỏ, ví dụ đốm than lá lúa.

---

## 7. Bước 3 – Tách lá

**Hàm:** `leaf_mask(rgb, config)`. Kết quả là mặt nạ nhị phân, True ở vị trí lá. Hàm dùng **RGB gốc**, không dùng ảnh đã tiền xử lý, để giữ quy ước màu cố định giữa các thí nghiệm so sánh.

### 7.1 ExG – Excess Green (chỉ số dư xanh lá)

$$\text{ExG}=2g-r-b=(g-r)+(g-b)=3g-1$$

**Ý nghĩa:** kênh xanh lá vượt hai kênh còn lại bao nhiêu.
- Lá xanh: ExG lớn, khoảng 0.68.
- Xám, trắng, đen: ExG = 0.
- Đỏ, tím, xanh dương: ExG âm.

| Pixel | RGB | ExG |
|---|---|---:|
| Lá xanh | 60,140,50 | 0.680 |
| Lá trong bóng râm | 30,70,25 | 0.680 |
| Giấy trắng | 220,220,220 | 0.000 |
| Đất nâu | 120,90,60 | 0.000 |
| Vết vàng | 180,170,60 | 0.244 |
| Xám hơi ánh xanh | 100,104,100 | 0.026 |

**Ngưỡng `exg > 0.04`** tương đương $g>0.347$: kênh G phải chiếm hơn 34.7% tổng màu, trong khi mốc trung tính là 33.3%. Biên 0.04 loại các pixel xám bị nhiễu JPEG hoặc ánh sáng làm hơi ngả xanh.

**Seed (hạt giống)**, tức các pixel *chắc chắn* là lá:

$$\text{seed}=(\text{ExG}>0.04)\wedge(S>0.15)$$

**Nhược điểm của ExG:** không phân biệt được đất nâu với giấy trắng (cả hai bằng 0). Vết bệnh nâu/đỏ có ExG thấp, nên nếu chỉ dùng ExG thì vết bệnh bị cắt khỏi lá.

### 7.2 Luật màu HSV

$$\text{color}=(8^\circ\le H\le170^\circ)\wedge(S>0.15)\wedge(V>0.06)$$

- Khoảng H **cố ý gồm cả vàng, cam, nâu** để vết bệnh vẫn thuộc về lá.
- $S>0.15$ loại nền trắng/xám.
- $V>0.06$ loại vùng gần đen, nơi H không ổn định.

**Nhược điểm:** đất nâu có H ≈ 30° và S ≈ 0.5, nên cũng lọt qua luật này.

### 7.3 Otsu – ngưỡng tự động – `otsu`

**Bài toán:** cho tập giá trị (ví dụ ExG của mọi pixel), tìm ngưỡng $t$ chia chúng thành 2 nhóm "tự nhiên" nhất.

**Cách làm:**
1. Quy giá trị về 256 bin và tính histogram $h(i)$, xác suất $p_i=h(i)/N$.
2. Với mỗi ngưỡng $t$:
$$w_0(t)=\sum_{i\le t}p_i,\quad w_1=1-w_0,\quad \mu_0=\frac{\sum_{i\le t}ip_i}{w_0},\quad \mu_1=\frac{\sum_{i>t}ip_i}{w_1}$$
3. **Phương sai giữa hai lớp**:
$$\sigma_B^2(t)=w_0\,w_1\,(\mu_0-\mu_1)^2$$
4. Chọn $t^*=\arg\max_t\sigma_B^2(t)$, rồi đổi về đơn vị gốc: $t_{gốc}=\min+t^*\cdot\frac{\max-\min}{255}$.

Code dùng dạng rút gọn tương đương, tính được bằng tổng tích lũy trong $O(256)$. Với $W(t)=\sum_{i\le t}h(i)$, $m(t)=\sum_{i\le t}i\,h(i)$, $N$ tổng số pixel, $M$ tổng mômen:
$$\sigma_B^2(t)\ \propto\ \frac{\big(M\,W(t)-m(t)\,N\big)^2}{W(t)\,\big(N-W(t)\big)}$$

**Tại sao:** cực đại $\sigma_B^2$ tương đương cực tiểu phương sai trong mỗi nhóm (vì tổng phương sai không đổi). Hai nhóm vì thế "gọn" nhất. Ngưỡng **tự thích nghi với từng ảnh**, không cần chọn tay.

**Hạn chế:** cần histogram có hai đỉnh rõ. Khi lá chiếm gần hết khung hình hoặc nền là cỏ xanh thì Otsu chia sai.

**Dùng ở đâu:** `exg_otsu`, `exg_exr` (tách lá) và `otsu_a` (vết bệnh). Cả ba là phương án đối chứng.

### 7.4 ExR và ExG − ExR

$$\text{ExR}=1.4r-g,\qquad \text{chỉ số}=\text{ExG}-\text{ExR}$$

ExR đo độ đỏ/nâu. Phép trừ đẩy đất nâu xuống âm sâu: lá ≈ +0.90, giấy ≈ −0.13, đất ≈ −0.29. Nhờ vậy lá và nền cách xa nhau hơn, Otsu dễ chia hơn. Kỹ thuật này phổ biến khi tách cây khỏi nền đất.

### 7.5 Các phương pháp tách lá (`leaf_method`)

| Giá trị | Mask ứng viên | Chọn mảng nào |
|---|---|---|
| **`hybrid`** (mặc định) | luật HSV | mảng chứa **nhiều seed nhất** |
| `hsv` | luật HSV | mảng lớn nhất |
| `exg_otsu` | ExG > Otsu(ExG) | mảng lớn nhất |
| `exg_exr` | (ExG−ExR) > Otsu(·) | mảng lớn nhất |

**Tại sao hybrid tốt:**
- HSV cho **hình dạng đầy đủ** của lá, kể cả vết bệnh vàng/nâu.
- Seed ExG cho **sự chắc chắn**: mảng đất nâu gần như không có seed nên bị loại.
- Nếu ảnh không có seed nào (lá úa vàng hết), hybrid quay về chọn mảng lớn nhất.

### 7.6 Hình thái học nhị phân – `morphology`, `cleanup`

Dùng phần tử cấu trúc (SE) $B$ hình chữ nhật $h\times w$, mặc định $3\times3$:

$$\textbf{Co (erosion)}:\ (A\ominus B)(p)=\bigwedge_{q\in B}A(p+q)\qquad \textbf{Giãn (dilation)}:\ (A\oplus B)(p)=\bigvee_{q\in B}A(p+q)$$

$$\textbf{Mở}:\ A\circ B=(A\ominus B)\oplus B\qquad \textbf{Đóng}:\ A\bullet B=(A\oplus B)\ominus B$$

- **Mở** xóa đốm nhiễu nhỏ hơn SE mà không làm co lá.
- **Đóng** lấp khe và lỗ nhỏ trong lá.

`cleanup` = mở rồi đóng. SE $1\times7$ hoặc $7\times1$ là biến thể để thử với lá lúa dài và mảnh.

**Đệm biên kiểu `edge`:** trong `morphology`, phần ngoài ảnh mặc định là False. Phép co vì vậy sẽ ăn mòn phần lá chạm mép ảnh. Code sao chép pixel biên ra ngoài $\max(h,w)+1$ ô trước khi xử lý, sau đó cắt bỏ phần đệm.

### 7.7 Gán nhãn thành phần liên thông – `components`

Thuật toán **loang (flood fill) bằng BFS**, liên thông 4 hướng:
1. Quét mọi pixel True chưa có nhãn.
2. Gán nhãn mới, rồi BFS lan sang các láng giềng trên, dưới, trái, phải.
3. Ghi lại diện tích mỗi mảng.

Độ phức tạp $O(HW)$. `areas[0]` là chỗ dành sẵn cho nền.

**Tại sao:** để tách mask thành từng "mảng" rời nhau rồi chọn đúng mảng lá. Cũng dùng để đếm và lọc vết bệnh.

### 7.8 Lấp lỗ – `fill_holes`

Gán nhãn liên thông trên **phần bù** $\neg A$. Mảng nền nào **không chạm mép ảnh** là lỗ nằm trong lá, và được tô thành True:

$$\text{fill}(A)=A\ \cup\ \{p\in\neg A : \text{mảng chứa } p \text{ không chạm biên ảnh}\}$$

**Tại sao:** vết bệnh đỏ thẫm (H < 8°) hoặc rất tối có thể bị luật HSV bỏ ra, tạo lỗ thủng. Lấp lỗ giúp diện tích lá đúng, nên tỷ lệ % tổn thương đúng.

**Kiểm tra cuối:** không tìm thấy mảng nào hoặc lá nhỏ hơn 40 px thì báo lỗi, yêu cầu chụp lại.

---

## 8. Bước 4 – Tìm vết bệnh

**Hàm:** `lesion_mask(processed, leaf, config, valid=~glare)`. Dùng **ảnh đã tiền xử lý**.

### 8.1 Vùng xét

$$\text{ROI}=\text{leaf}\wedge\neg\text{glare}$$

Hàm lấy ngẫu nhiên tối đa **8000 pixel** của ROI (seed 42) để chạy K-means cho nhanh. Sau đó gán cụm cho **toàn bộ** pixel ROI.

### 8.2 K-means với khởi tạo K-means++ – `kmeans`

**Mục tiêu:** chia $n$ điểm $x_i$ thành $K$ cụm, cực tiểu tổng bình phương khoảng cách tới tâm (SSE):

$$J=\sum_{i=1}^{n}\lVert x_i-\mu_{c(i)}\rVert^2$$

**Khởi tạo K-means++:**
1. Tâm đầu tiên chọn ngẫu nhiên đều.
2. Mỗi tâm tiếp theo chọn điểm $x$ với xác suất tỷ lệ với $D(x)^2$, trong đó $D(x)$ là khoảng cách tới tâm gần nhất đã chọn:
$$P(x)=\frac{D(x)^2}{\sum_{x'}D(x')^2}$$

**Vòng lặp Lloyd** (tối đa 60 vòng; dừng khi tâm dịch chuyển dưới $10^{-6}$):
$$c(i)=\arg\min_j\lVert x_i-\mu_j\rVert^2,\qquad \mu_j=\frac{1}{|C_j|}\sum_{i\in C_j}x_i$$

Cụm rỗng giữ tâm cũ.

**Tại sao K-means++:** khởi tạo dàn đều các tâm, tránh hai tâm rơi vào cùng một vùng màu. Kết quả ổn định hơn và hội tụ nhanh hơn khởi tạo ngẫu nhiên thuần.

**Tại sao phân cụm thay vì đặt ngưỡng màu cố định:** màu lá khỏe khác nhau giữa lúa và cà phê, giữa các ảnh và các điều kiện sáng. Phân cụm **so sánh tương đối trong chính lá đó**.

### 8.3 Quyết định cụm nào là vết bệnh (`lesion_method`)

**`ab_distance` (mặc định)**, phân cụm trên $(a^*,b^*)$ với $K=3$:
1. **Cụm khỏe** $h$ = cụm có **nhiều pixel nhất**. Giả định: phần lớn lá còn khỏe.
2. Khoảng cách từng tâm tới tâm khỏe:
$$d_j=\lVert\mu_j-\mu_h\rVert_2\ \ (\text{trong mặt phẳng } a^*b^*)$$
3. Cụm **xa nhất** $j^*=\arg\max_j d_j$ là vết bệnh **chỉ khi** $d_{j^*}\ge 8$ (`lesion_min_distance`). Ngược lại thì lá không có vết bệnh.

Lý do:
- **Bỏ L\***: bóng râm và phản sáng chủ yếu thay đổi L\*, không phải màu.
- **Ngưỡng 8**: lá đồng màu vẫn bị K-means chia ra 3 cụm (thuật toán *luôn* chia được). Ngưỡng này ngăn việc "bịa" ra vết bệnh. Mốc 8 tương ứng một khác biệt màu thấy rõ bằng mắt.
- **Chỉ lấy một cụm**: cụm ở giữa thường là vùng chuyển tiếp hoặc gân lá. Chọn một cụm xa nhất là cách bảo thủ.

**`lab_rule`** (luật cũ, đối chứng), phân cụm trên $(L^*,a^*,b^*)$. Cụm $j$ là vết bệnh nếu:
$$(a^*_j>-3\ \wedge\ b^*_j>8)\ \vee\ L^*_j<30$$
Nghĩa là "bớt xanh và ngả vàng", hoặc "rất tối".

**`otsu_a`** (đối chứng): $\text{lesion}=a^*>\text{Otsu}(a^*)$. Kênh a\* càng dương thì càng bớt xanh.

### 8.4 Hậu xử lý

- `lesion_close` (mặc định tắt): đóng hình thái $3\times3$ để nối các đốm vụn.
- Gán nhãn liên thông và **loại mảng nhỏ hơn 3 pixel** (`min_lesion_pixels`) để bỏ nhiễu điểm.
- `centers`: tâm $(L^*,a^*,b^*)$ trung bình của mỗi cụm, hiển thị ở tab Giải thích.

---

## 9. Bước 5 – Tỷ lệ diện tích, mức độ, cảnh báo

**Tỷ lệ diện tích nghi tổn thương:**

$$\text{ratio}=\frac{|\text{lesion}|}{|\text{leaf}|}$$

Tỷ lệ này tính trên **diện tích chiếu của phần lá nhìn thấy** trong ảnh, không phải toàn bộ lá thật.

**Mức độ** – `severity()` đọc `data/severity_rules.json`. Ngưỡng $[0.01,\ 0.05,\ 0.20]$, khoảng **đóng bên trái**:

| ratio | Mức |
|---|---|
| < 1% | Chưa thấy vùng nghi tổn thương |
| 1% – <5% | Nhẹ |
| 5% – <20% | Trung bình |
| ≥ 20% | Nặng |

Code dùng `np.searchsorted(thresholds, ratio, side='right')`. Đây là **quy ước kỹ thuật của đồ án, chưa hiệu chuẩn nông học**. Thang IRRI chưa được ánh xạ (`irri_score = null`).

**Cảnh báo chất lượng** (`process`):

| Điều kiện | Ý nghĩa |
|---|---|
| lá < 3% hoặc > 90% ảnh | mask lá có thể sai |
| lá chạm biên ảnh | tỷ lệ chỉ phản ánh phần nhìn thấy |
| chói > 3% diện tích lá | phần chói bị bỏ, tỷ lệ có thể bị đánh giá thấp |
| độ lệch chuẩn RGB < 8 | ảnh thiếu tương phản, nên chụp lại |

**Overlay:** pixel vết bệnh được trộn 50% với màu đỏ $(255,40,40)$ để hiển thị.

---

## 10. Bước 6 – Trích 403 đặc trưng

**Hàm:** `features(processed, leaf, lesion, valid, raw)` trong `features.py`. Vector luôn có đúng 403 chiều, kiểm tra hữu hạn (không NaN/inf).

| Khối | Nhóm (`FEATURE_GROUPS`) | Số chiều | Phụ thuộc mask vết bệnh? |
|---|---|---:|---|
| Vùng tổn thương | `color` | 102 | có |
| | `texture` | 26 | có |
| | `shape` | 7 | có |
| | `lesion` = 3 nhóm trên | 135 | có |
| Toàn lá | `leaf` | 268 | **không** |
| Tất cả | `all` | 403 | |

**Tại sao có hai khối:**
- Khối tổn thương mô tả **chính vết bệnh**.
- Khối toàn lá **không phụ thuộc chất lượng mask vết bệnh**. Nếu bước 4 tìm sai vết, khối này vẫn mang thông tin đúng.

Thực nghiệm cho thấy CV chọn nhóm `leaf` cho cả hai cây (xem mục 15).

### 10.1 Khối tổn thương (135) – `lesion_features`

$\text{roi}=\text{lesion}\wedge\text{leaf}\wedge\text{valid}$. Nếu $|\text{roi}|/|\text{leaf}|<1\%$ thì **cả 135 giá trị = 0**, vì vết quá nhỏ cho đặc trưng không tin cậy.

**a) Màu – 102 chiều**
- **Histogram** 16 bin cho mỗi kênh, chuẩn hóa tổng bằng 1. HSV dùng khoảng $H\in[0,360]$, $S,V\in[0,1]$. Lab dùng khoảng $L\in[0,100]$, $a,b\in[-128,128]$. Tổng $6\times16=96$:
$$\text{hist}_k=\frac{\#\{p\in\text{roi}: v(p)\in\text{bin }k\}}{|\text{roi}|}$$
- **Mômen màu**: trung bình và độ lệch chuẩn của $L^*,a^*,b^*$ trong ROI, thêm $6$ chiều.

Lý do: mỗi loại bệnh có màu đặc trưng. Gỉ sắt màu cam, đốm than màu đen, bạc lá màu vàng rơm. Histogram mô tả **phân bố** màu, mômen mô tả **tâm và độ tản**.

**b) Kết cấu – 26 chiều.** Ảnh xám: $\text{gray}=0.299R+0.587G+0.114B$ (chuẩn ITU-R BT.601).

**GLCM** (Gray-Level Co-occurrence Matrix, ma trận đồng xuất hiện mức xám) – `glcm`, `glcm_properties`:
1. Lượng tử ảnh xám về $L=16$ mức: $q=\lfloor\text{gray}\cdot16/256\rfloor$.
2. Với độ dời $(d_y,d_x)$, đếm số cặp pixel $(p,\ p+(d_y,d_x))$ cùng nằm trong ROI có mức $(i,j)$:
$$C(i,j)=\#\{p: q(p)=i,\ q(p+\delta)=j\}$$
3. Làm **đối xứng** và **chuẩn hóa**: $P=\dfrac{C+C^\top}{\sum(C+C^\top)}$.
4. Sáu thuộc tính Haralick, với $\mu_i=\sum iP$, $\sigma_i^2=\sum(i-\mu_i)^2P$ (tương tự cho $j$):

| Thuộc tính | Công thức | Ý nghĩa |
|---|---|---|
| Contrast | $\sum_{i,j}(i-j)^2P(i,j)$ | độ chênh mạnh giữa pixel kề |
| Dissimilarity | $\sum\lvert i-j\rvert P(i,j)$ | như contrast nhưng tuyến tính |
| Homogeneity | $\sum\dfrac{P(i,j)}{1+(i-j)^2}$ | độ đồng nhất |
| ASM | $\sum P(i,j)^2$ | độ "trật tự", ít mẫu lặp |
| Energy | $\sqrt{\text{ASM}}$ | |
| Correlation | $\dfrac{\sum(i-\mu_i)(j-\mu_j)P}{\sigma_i\sigma_j}$ | tương quan tuyến tính giữa pixel kề |

Thuộc tính được lấy **trung bình trên 4 hướng** $(0,1),(1,0),(1,1),(-1,1)$, tức 0°, 90°, 45°, 135°, để không phụ thuộc hướng xoay. Kết quả: 6 chiều.

**LBP** (Local Binary Pattern, mẫu nhị phân cục bộ) – `lbp8`. Với tâm $g_c$ và 8 láng giềng $g_p$ trên **hình vuông** bán kính $R$ (theo thứ tự E, NE, N, NW, W, SW, S, SE):
$$\text{LBP}_R=\sum_{p=0}^{7}s(g_p-g_c)\,2^p,\qquad s(x)=\begin{cases}1&x\ge0\\0&x<0\end{cases}$$

Ánh xạ **uniform**: đếm số lần chuyển 0↔1 theo vòng, ký hiệu $U$. Nếu $U\le2$ (mẫu "đều": điểm, cạnh, góc, vùng phẳng) thì nhãn = số bit 1, từ 0 đến 8. Ngược lại nhãn = 9. Tổng cộng **10 bin**.

Histogram chuẩn hóa, chỉ lấy tại các tâm mà cả cửa sổ $(2R+1)^2$ nằm trong ROI (phép co ROI). $R\in\{1,2\}$ cho 20 chiều.

Lý do dùng hai bộ kết cấu:
- **GLCM** mô tả thống kê bậc hai của độ xám, phân biệt bề mặt mịn với sần.
- **LBP** mô tả vi cấu trúc cục bộ và **bất biến với thay đổi độ sáng đơn điệu**, vì chỉ so sánh lớn hay bé.

Ví dụ: đốm than là chấm nhỏ dày, bạc lá là vệt dài mịn.

**c) Hình dạng – 7 chiều** (liên thông 4 hướng trên ROI):

| # | Đặc trưng | Công thức |
|---|---|---|
| 1 | `lesion_count` | số mảng vết |
| 2 | `mean_area_fraction` | $\overline{A_k}/\lvert\text{leaf}\rvert$ |
| 3 | `max_area_fraction` | $\max A_k/\lvert\text{leaf}\rvert$ |
| 4 | `mean_eccentricity` | trung bình độ lệch tâm (bên dưới) |
| 5 | `mean_solidity` | trung bình độ đặc (bên dưới) |
| 6 | `edge_distance_normalized` | $\overline{D(p)}_{p\in\text{roi}}/\sqrt{\lvert\text{leaf}\rvert}$ |
| 7 | `lesion_ratio` | $\lvert\text{roi}\rvert/\lvert\text{leaf}\rvert$ |

- **Độ lệch tâm** (`region_shape`): từ ma trận hiệp phương sai tọa độ pixel $\Sigma=\frac1n\sum(p-\bar p)(p-\bar p)^\top$, với trị riêng $\lambda_1\le\lambda_2$:
$$e=\sqrt{1-\lambda_1/\lambda_2}$$
  $e\approx0$ là vết tròn (đốm nâu), $e\to1$ là vết dài (bạc lá).
- **Độ đặc (solidity)**: $\dfrac{\text{diện tích}}{\text{diện tích bao lồi}}$. Bao lồi được tính bằng **thuật toán monotone chain của Andrew**: sắp xếp điểm, dựng chuỗi dưới và chuỗi trên, loại điểm khi tích có hướng $\le0$, độ phức tạp $O(n\log n)$. Vết tròn đầy có solidity ≈ 1; vết loang lổ, răng cưa có solidity thấp.
- **Biến đổi khoảng cách Euclid** (`distance_transform`, thuật toán Felzenszwalb–Huttenlocher): với mỗi pixel lá, $D(p)$ là khoảng cách Euclid tới pixel nền gần nhất. Thuật toán tách thành 2 lượt 1D (theo hàng rồi theo cột), mỗi lượt tính **bao dưới của các parabol** $(q-v)^2+f(v)$ trong $O(n)$. Đặc trưng #6 cho biết vết nằm **gần mép** (nhỏ, ví dụ bạc lá lan từ mép) hay **giữa phiến lá** (lớn). Chia cho $\sqrt{|\text{leaf}|}$ để không phụ thuộc kích thước ảnh.

### 10.2 Khối toàn lá (268) – `leaf_features`

Mọi histogram ở khối này lấy **căn bậc hai** sau khi chuẩn hóa, gọi là **ánh xạ Hellinger**:

$$\phi(h)=\sqrt{h/\textstyle\sum h}\quad\Rightarrow\quad \lVert\phi(h)-\phi(h')\rVert_2^2=2-2\sum_k\sqrt{h_kh'_k}$$

Khoảng cách Euclid sau ánh xạ chính là (lần) **khoảng cách Hellinger** giữa hai phân bố. Nhờ đó vài bin rất lớn (ví dụ "màu xanh lá khỏe") không lấn át các bin nhỏ nhưng mang thông tin (màu vết bệnh). Cách này phù hợp với k-NN và kernel dùng khoảng cách Euclid.

| Thành phần | Ảnh dùng | Chi tiết | Số chiều |
|---|---|---|---:|
| Histogram 2D a\*b\* | đã tiền xử lý | $10\times10$ bin, $a\in[-50,50]$, $b\in[-30,70]$, trên toàn lá, bỏ L\* | 100 |
| LBP uniform | **gốc** | 3 kênh L\*, a\*, b\* × $R\in\{1,2,3\}$ × 10 bin | 90 |
| Histogram độ lớn gradient | **gốc** | 3 kênh × 8 bin, trên lá đã co $3\times3$ | 24 |
| GLCM | **gốc** | 3 kênh × khoảng cách $\{1,2,4\}$ × 6 thuộc tính (trung bình 4 hướng) | 54 |

Chi tiết:
- **Lượng tử hóa kênh Lab về 8 bit** trước LBP/GLCM: $L\in[0,100]$, $a,b\in[-64,64]$, ánh xạ tuyến tính về $[0,255]$.
- **Gradient**: $\nabla I=(\partial_yI,\partial_xI)$ bằng **sai phân trung tâm** (`np.gradient`), độ lớn $\lvert\nabla I\rvert=\sqrt{(\partial_xI)^2+(\partial_yI)^2}$. Histogram trong khoảng $[0,30]$ cho L\* và $[0,20]$ cho a\*, b\*. Gradient mô tả mật độ biên, đường gân và mép vết bệnh.
- **Nhiều bán kính và khoảng cách** (LBP 1, 2, 3; GLCM 1, 2, 4) giúp bắt được kết cấu ở **nhiều thang kích thước**: chấm nhỏ và mảng lớn.
- **Tại sao dùng ảnh gốc cho kết cấu:** CLAHE làm thay đổi kết cấu cục bộ (tăng nhiễu vùng phẳng). Màu thì lấy từ ảnh đã cân bằng trắng để ổn định với ánh sáng.

---

## 11. Bước 7 – Phân lớp

Tất cả bộ phân lớp nằm trong `model.py` và tự cài bằng NumPy.

### 11.1 Chọn nhóm đặc trưng và chuẩn hóa z-score

Mô hình chỉ dùng các cột `indices` thuộc một nhóm (`leaf`, `all`, ...). Mean và độ lệch chuẩn được tính **chỉ trên tập train**:

$$z_j=\frac{x_j-\mu_j}{\sigma_j}\qquad(\sigma_j<10^{-8}\Rightarrow\sigma_j:=1)$$

**Tại sao:** k-NN, LDA và RBF đều dựa trên khoảng cách. Không chuẩn hóa thì đặc trưng có thang lớn (ví dụ `lesion_count`) sẽ lấn át đặc trưng thang nhỏ (bin histogram). Chỉ fit trên train để **không rò rỉ thông tin từ test**.

### 11.2 k-NN – k láng giềng gần nhất (`KNN`)

**Cách hoạt động:**
1. Tính khoảng cách Euclid từ mẫu mới $z$ tới mọi mẫu train: $d_i=\lVert z-z_i\rVert_2$.
2. Lấy $k$ mẫu gần nhất (sắp xếp ổn định). Mỗi mẫu bỏ **một phiếu ngang nhau**.
3. Nhãn = lớp nhiều phiếu nhất. Khi hòa, chọn lớp đứng trước theo thứ tự chữ cái.
4. $\text{vote}=\dfrac{\text{số phiếu của lớp thắng}}{k}$.

**Tại sao:** k-NN không có tham số để học, đơn giản, dễ giải thích ("ảnh này giống 3 ảnh train sau"), và hợp với dữ liệu nhỏ. Hạn chế là nhạy với đặc trưng nhiễu và chiều cao, nên mới cần thêm LDA.

### 11.3 LDA có co ma trận (Shrinkage LDA) + k-NN – `lda_projection`

**Ý tưởng (Fisher):** tìm hướng chiếu sao cho **các lớp cách xa nhau** và **mỗi lớp co cụm**.

- Ma trận tán xạ trong lớp và giữa các lớp, với $\mu_c$ là trung bình lớp $c$, $\mu$ trung bình chung:
$$S_W=\frac1N\sum_c\sum_{i\in c}(z_i-\mu_c)(z_i-\mu_c)^\top,\qquad S_B=\frac1N\sum_c N_c(\mu_c-\mu)(\mu_c-\mu)^\top$$
- **Tiêu chí Fisher**:
$$\max_v\ \frac{v^\top S_Bv}{v^\top S_Wv}\ \Longleftrightarrow\ S_Bv=\lambda S_Wv$$
- **Co (shrinkage)** với hệ số $s$, gọi là `lda` ∈ {0.03, 0.1, 0.3}:
$$\tilde S_W=(1-s)S_W+s\,\frac{\operatorname{tr}(S_W)}{d}I$$
  Lý do: với 268 chiều và vài chục đến vài trăm mẫu mỗi lớp, $S_W$ gần **suy biến** nên không nghịch đảo ổn định được. Kéo về ma trận đường chéo làm $\tilde S_W$ khả nghịch và giảm overfit.
- **Giải bằng làm trắng (whitening):**
  1. Phân rã $\tilde S_W=U\Lambda U^\top$ và đặt $W=U\Lambda^{-1/2}$, nên $W^\top\tilde S_WW=I$.
  2. Giải bài toán trị riêng thường $(W^\top S_BW)a=\lambda a$.
  3. Ma trận chiếu $V=W\,[a_1\ \dots\ a_{C-1}]$, sắp theo $\lambda$ giảm dần, thỏa $v^\top\tilde S_Wv=1$.
- $S_B$ có hạng tối đa $C-1$, nên chỉ có **$C-1$ trục** có ý nghĩa: 4 trục cho cà phê, 2 trục cho lúa.

Sau đó chạy k-NN trong không gian $z\,V$, có $C-1$ chiều.

**Tại sao:** giảm từ 268 chiều xuống 4 chiều **có định hướng theo nhãn**. Khoảng cách Euclid trong không gian LDA phản ánh sự khác biệt giữa các bệnh thay vì nhiễu. Với cà phê, cách này nâng accuracy từ 63.9% (v2) lên 88.4%.

### 11.4 Kernel Ridge với nhân RBF – `KernelRidge`

**Nhân RBF (Gaussian)** – `rbf_kernel`:
$$K(z,z')=\exp\!\left(-\gamma_{eff}\lVert z-z'\rVert^2\right),\qquad \gamma_{eff}=\frac{\gamma}{d}$$
Chia cho số chiều $d$ để thang $\gamma$ không phụ thuộc số đặc trưng.

**Một-chọi-phần-còn-lại (one-vs-rest):**
- Đích của mỗi lớp: $T_{ic}=+1$ nếu mẫu $i$ thuộc lớp $c$, ngược lại $-1$.
- Trừ đi trung bình cột: $\text{offset}_c=\overline{T_{\cdot c}}$.

**Hồi quy ridge trong không gian nhân** (dạng đối ngẫu) – `kernel_ridge_fit`:
$$\min_f\ \sum_i\lVert f(z_i)-t_i\rVert^2+\alpha\lVert f\rVert_{\mathcal H}^2\ \Longrightarrow\ (K+\alpha I)\,W=T-\text{offset}$$

**Dự đoán:**
$$f(z)=\big[K(z,z_1)\ \dots\ K(z,z_N)\big]\,W+\text{offset},\qquad \hat y=\arg\max_c f_c(z)$$

**Ý nghĩa tham số:**
- $\gamma$ lớn: nhân hẹp, mô hình "nhìn gần", dễ overfit. $\gamma$ nhỏ: biên quyết định trơn.
- $\alpha$ lớn: phạt mạnh, mô hình đơn giản hơn. $\alpha$ nhỏ: khớp sát train.

**Tại sao:** là bộ phân lớp **phi tuyến** có nghiệm đóng (chỉ cần giải một hệ tuyến tính), không cần tối ưu lặp như SVM, dễ tự cài bằng NumPy. Với 90 ảnh lúa, ma trận $90\times90$ giải tức thì. Đây là cấu hình CV chọn cho lúa.

### 11.5 Cơ chế từ chối (rejection)

Mô hình **không bắt buộc phải trả lời**. Một ảnh bị từ chối khi:

$$\text{rejected}=(\text{vote}<0.6)\ \vee\ (d_{min}>\text{radius})$$

- **vote**:
  - Với k-NN: tỷ lệ phiếu của lớp thắng.
  - Với Kernel Ridge: tỷ lệ trong $k=5$ láng giềng gần nhất **đồng ý với nhãn ridge dự đoán**.
  - Vote dưới 60% nghĩa là các láng giềng không thống nhất.
- **radius** (bán kính "trong miền dữ liệu"), trong đó $d^{NN}_i$ là khoảng cách từ mẫu train $i$ tới mẫu train gần nhất khác nó:
$$\text{radius}=1.5\times Q_{0.95}\big(\{d^{NN}_i\}\big)$$
  Ảnh mới xa hơn mức này nghĩa là **khác hẳn mọi thứ đã học**, ví dụ ảnh không phải lá hay cây trồng khác, nên bị từ chối.
- **Riêng với LDA** – `_held_out_nearest`: LDA được fit bằng nhãn train nên các mẫu train co cụm *giả tạo* trong không gian LDA, làm radius quá nhỏ và từ chối gần hết ảnh mới. Vì vậy $d^{NN}_i$ được đo theo **5 fold phân tầng**: chiếu mẫu $i$ bằng LDA fit trên các fold không chứa $i$, rồi đo tới các mẫu đó.

**Lưu ý:** vote **không phải xác suất** đã hiệu chuẩn. App ghi rõ điều này cho người dùng.

### 11.6 Giải thích – `explain`

- **3 mẫu train gần nhất**: id, nhãn, khoảng cách.
- **5 thành phần đóng góp nhiều nhất** vào $\lVert z-z_{nn}\rVert^2$, là các số hạng $(z_j-z_{nn,j})^2$ lớn nhất. Khi dùng LDA, các thành phần này là các trục `lda_axis_k`.

Đây là **phân rã khoảng cách**, không phải độ quan trọng nhân quả.

---

## 12. Bước 8 – Khuyến nghị

`recommendation(label, ratio, rejected)`:

```
nếu bị từ chối hoặc không có mô hình  → câu "uncertain": chụp lại, hỏi cán bộ khuyến nông
ngược lại:
    câu riêng theo bệnh (diseases[label])
  + câu chung: mức ≤ Nhẹ → "low" (đánh dấu, theo dõi)
               mức ≥ Trung bình → "high" (khảo sát thêm, liên hệ bảo vệ thực vật)
```

Nguyên tắc: **không kê thuốc** và không suy tác nhân chỉ từ màu. Mọi nội dung nằm trong `data/severity_rules.json`, nên sửa câu chữ không cần sửa code.

---

## 13. Huấn luyện và đánh giá

### 13.1 Kiểm tra dữ liệu – `read_manifest`

Mỗi dòng manifest gồm: `id, path, crop, label, domain (plain/field), group, split (train/test/segval/segtest), leaf_mask, lesion_mask, source, license`.

Các kiểm tra:
- id không trùng; label và group không rỗng.
- crop, domain, split hợp lệ.
- **Một group (cây, phiên chụp) không được nằm ở hai split khác nhau**. Đây là chống **rò rỉ dữ liệu**: ảnh chụp liên tiếp cùng một lá mà rơi vào cả train lẫn test sẽ làm điểm số ảo cao.
- **Không có hai file trùng nội dung**, so bằng băm SHA-256.
- File ảnh và mask phải tồn tại.

### 13.2 Kiểm định chéo phân nhóm – `StratifiedGroupKFold`

- **Số fold** $=\min(5,\ \text{số group ít nhất của một lớp})$, có xáo trộn, `random_state=42`.
- **Stratified**: mỗi fold giữ tỷ lệ lớp gần giống toàn tập.
- **Group**: cùng một group luôn nằm trọn trong một fold.
- **Chỉ dùng tập train**. Tập test bị khóa cho đến khi đã chọn xong mô hình.
- Mỗi fold phải có đủ mọi lớp, nếu không thì báo lỗi.

**Lưới siêu tham số** (tổng 24 + 24 + 18 = 66 cấu hình mỗi cây):

| Phương pháp | Nhóm đặc trưng | Tham số |
|---|---|---|
| `knn` | color, texture, shape, lesion, leaf, all | $k\in\{1,3,5,7\}$ |
| `lda+knn` | leaf, all | shrinkage $\in\{0.03,0.1,0.3\}$, $k\in\{1,5,11,21\}$ |
| `kernel_ridge` | leaf, all | $\gamma\in\{0.1,0.3,1\}$, $\alpha\in\{0.01,0.1,1\}$, $k=5$ |

Chọn cấu hình có **macro-F1 trung bình qua các fold** cao nhất, fit lại trên toàn bộ train, rồi chấm test **một lần**.

### 13.3 Chỉ số đánh giá – `classification`

Từ **ma trận nhầm lẫn** $CM[\text{thật},\text{dự đoán}]$. Ảnh test mà tách lá thất bại được ghi vào một **cột riêng** `processing_failed` và luôn tính là sai.

$$\text{Accuracy}=\frac{\sum_c TP_c}{N}$$

$$F1_c=\frac{2\,TP_c}{2TP_c+FP_c+FN_c}=\frac{2\,TP_c}{\text{tổng cột }c+\text{tổng hàng }c},\qquad \text{macro-F1}=\frac1C\sum_cF1_c$$

**Tại sao chọn theo macro-F1:** dữ liệu mất cân bằng lớp. Accuracy có thể cao chỉ nhờ đoán đúng lớp đông. Macro-F1 cho **mỗi lớp trọng số như nhau**, nên lớp hiếm như cercospora cũng được tính đủ.

**Các chỉ số bổ sung:**
- `coverage` = tỷ lệ ảnh test **không bị từ chối**.
- `accepted_accuracy` = accuracy trên các ảnh được chấp nhận.

Hai chỉ số này cho thấy cơ chế từ chối có tác dụng hay không.

### 13.4 Đánh giá mask – `overlap`, `evaluate_masks`

Áp dụng khi có mask vẽ tay ở split `segtest`:

$$\text{IoU}=\frac{|A\cap B|}{|A\cup B|},\qquad \text{Dice}=\frac{2|A\cap B|}{|A|+|B|}$$

Quy ước: cả hai rỗng thì bằng 1; chỉ một bên rỗng thì bằng 0. Hàm còn báo `leaf_iou` và sai số tuyệt đối của tỷ lệ diện tích.

**Trạng thái hiện tại: `pending`**, vì chưa có đủ 30 mask tay, nên chưa có số IoU thật.

### 13.5 Đối chứng và kiểm tra độ bền

- **SVM RBF** (sklearn, chỉ để so sánh):
  - Pipeline: `StandardScaler → PCA (tắt hoặc giữ 95% phương sai) → SVC(class_weight='balanced')`.
  - Lưới: $C\in\{0.1,1,10\}$, $\gamma\in\{\text{scale},0.01\}$.
  - Dùng cùng các fold. Scaler và PCA fit **bên trong** từng fold.
- **GrabCut** (OpenCV, `scripts/reference_methods.py`): bộ tách lá đối chứng, dựa trên mô hình hỗn hợp Gauss và cắt đồ thị (graph cut).
- **Stress test**: giữ nguyên mô hình, biến đổi ảnh test:
  - Độ sáng ×0.6, ×1.4.
  - Làm mờ Gauss với $\sigma=1,2$.
  - Xoay 90°, 180°, 15°.

  `end_to_end_accuracy` giữ **đủ mẫu số**: ảnh tách lá thất bại vẫn tính là sai.
- **Ablation tiền xử lý** (trên train/CV): không tiền xử lý, có tiền xử lý, luật Lab cũ. So sánh bằng k-NN k=3 trên nhóm `lesion`.
- **So sánh tách lá**: hybrid, HSV, ExG+Otsu, ExG−ExR, SE 1×7, SE 7×1, GrabCut.
- **So sánh tách vết bệnh**: `ab_distance` và `lab_rule` với K = 2, 3, 4; `otsu_a`.

### 13.6 File kết quả

| File | Nội dung |
|---|---|
| `results/metrics.json` | Mọi số liệu: cấu hình chọn, test, SVM, stress, ablation, mask |
| `results/cross_validation.json` | Điểm từng fold của từng cấu hình trong lưới |
| `results/predictions.csv` | Từng ảnh test: thật, dự đoán, vote, distance, rejected |
| `results/verification.json` | Đối chiếu số các thuật toán tự cài với thư viện chuẩn (`scripts/verify_algorithms.py`) |
| `results/environment.json` | Phiên bản OS, Python, thư viện |
| `results/*_confusion.png`, `pipeline_examples.png` | Hình minh họa |

---

## 14. Lưu và nạp mô hình

**`save`** dùng `np.savez_compressed` và **chỉ lưu mảng số/chuỗi, không dùng pickle**, nên nạp file không thể chạy mã độc. Nội dung file:
- `method`: `knn` hoặc `kernel_ridge`.
- `x`, `y`: dữ liệu train đã chuẩn hóa hoặc đã chiếu.
- `mean`, `std`, `indices`, `k`, `radius`, `lda`, `projection`.
- Với Kernel Ridge: thêm `alpha`, `gamma`, `weights`, `offset`.
- `config`: toàn bộ `Config` dưới dạng JSON.
- `crop`, `schema = "leaflab-3"`, `feature_dim = 403`, `feature_names`, `sample_ids`.

**`load`** nạp với `allow_pickle=False` và kiểm tra:
- Schema, số chiều và **tên từng đặc trưng** phải khớp. Model cũ bị từ chối kèm hướng dẫn chạy lại, thay vì chạy âm thầm với vector sai.
- Kích thước mọi mảng nhất quán.
- Giá trị hữu hạn, `std > 0`, `indices` nằm trong [0, 403).

---

## 15. Kết quả hiện tại và giới hạn

| Cây | Cấu hình CV chọn | macro-F1 CV | Test accuracy | Test macro-F1 | Coverage |
|---|---|---:|---:|---:|---:|
| Cà phê | nhóm `leaf`, LDA shrinkage 0.1 + k-NN $k=21$ | 0.820 ± 0.010 | **88.4%** | 0.841 | 89.3% |
| Lúa | nhóm `leaf`, Kernel Ridge $\gamma=0.1$, $\alpha=0.01$ | 0.887 ± 0.069 | **90.0%** | 0.900 | 83.3% |

Phiên bản v2 (135 đặc trưng, k-NN) đạt 63.9% cho cà phê và 63.3% cho lúa.

**Giới hạn cần biết:**
- **Group được chia theo ảnh**, vì nguồn dữ liệu không có ID cây hay phiên chụp. Ở BRACOL, riêng màu nền đã đoán được nhãn khoảng 69%. Khi giữ các ảnh có số thứ tự liền nhau ở cùng một phía, accuracy cà phê giảm còn khoảng 75–81%.
- **Test lúa chỉ có 30 ảnh**: mỗi ảnh tương ứng 3.3 điểm phần trăm, và tập này đã được nhìn nhiều lần trong quá trình phát triển. Chênh lệch giữa các bộ phân lớp nằm trong độ nhiễu.
- **Cercospora yếu**: đúng 16/34.
- **Độ bền**: accuracy cà phê còn 65–68% khi đổi độ sáng ×0.6/×1.4, và 44% khi mờ $\sigma=2$.
- **Chưa có ảnh tự chụp ngoài đồng** (yêu cầu ≥100) và chưa có mask tay (yêu cầu ≥30), nên chưa đo được độ chính xác thực tế và IoU.
- **Ngưỡng mức độ** là quy ước của đồ án; chưa ánh xạ thang IRRI.
- **Dữ liệu lúa không có lớp lá khỏe.**

---

## 16. Bảng tra cứu tham số

### `Config` (`src/pipeline.py`)

| Tham số | Mặc định | Dùng ở | Ý nghĩa |
|---|---|---|---|
| `max_side` | 256 | `load_rgb` | Cạnh dài tối đa sau khi thu nhỏ |
| `clusters` | 3 | `lesion_mask` | Số cụm $K$ của K-means |
| `seed` | 42 | K-means, lấy mẫu | Hạt ngẫu nhiên để tái lập kết quả |
| `exg` | 0.04 | `leaf_mask` | Ngưỡng ExG của seed ($g>34.7\%$) |
| `saturation` | 0.15 | `leaf_mask` | Ngưỡng S cho seed và luật màu |
| `lesion_a` | −3 | `lab_rule` | Ngưỡng a\* của tâm cụm bệnh |
| `lesion_b` | 8 | `lab_rule` | Ngưỡng b\* của tâm cụm bệnh |
| `dark_l` | 30 | `lab_rule` | Tâm cụm có L\* < 30 thì là vết tối |
| `preprocessing` | True | `preprocess` | Bật/tắt toàn bộ tiền xử lý |
| `white_balance` | True | `preprocess` | Gray-world |
| `clahe_clip` | 2.0 | `clahe` | Giới hạn bin, tính theo bội số diện tích ô / 256 |
| `clahe_tiles` | 8 | `clahe` | Lưới 8×8 ô |
| `median` | False | `preprocess` | Lọc trung vị 3×3 |
| `exclude_glare` | True | `process` | Loại pixel chói khỏi phân tích vết bệnh |
| `leaf_method` | `hybrid` | `leaf_mask` | `hybrid` / `hsv` / `exg_otsu` / `exg_exr` |
| `morph_height`, `morph_width` | 3, 3 | `cleanup` | Kích thước SE |
| `lesion_method` | `ab_distance` | `lesion_mask` | `ab_distance` / `lab_rule` / `otsu_a` |
| `lesion_min_distance` | 8.0 | `ab_distance` | Khoảng cách a\*b\* tối thiểu giữa cụm bệnh và cụm khỏe |
| `min_lesion_pixels` | 3 | `lesion_mask` | Bỏ mảng vết bệnh nhỏ hơn |
| `lesion_close` | False | `lesion_mask` | Đóng hình thái mask vết bệnh |

### Hằng số cố định trong code

| Giá trị | Ở đâu | Ý nghĩa |
|---|---|---|
| V ≥ 0.98, S ≤ 0.10 | `preprocess` | Định nghĩa pixel chói |
| gain ∈ [0.5, 2] | `gray_world` | Giới hạn hệ số cân bằng trắng |
| H ∈ [8°, 170°], V > 0.06 | `leaf_mask` | Luật màu lá |
| 1.4 | ExR | Hệ số kênh đỏ |
| 40 px | `leaf_mask` | Diện tích lá tối thiểu |
| 8000 | `lesion_mask` | Số pixel mẫu cho K-means |
| 60 vòng, 1e−6 | `kmeans` | Điều kiện dừng |
| 1% | `lesion_features` | Vết nhỏ hơn thì khối tổn thương bằng 0 |
| 16 mức / 16 bin | GLCM, histogram tổn thương | Lượng tử hóa |
| 10×10 | histogram a\*b\* toàn lá | |
| [0.01, 0.05, 0.20] | `severity_rules.json` | Ngưỡng mức độ |
| vote < 0.6 | `predict_details` | Từ chối do láng giềng không thống nhất |
| 1.5 × Q₀.₉₅ | `fit` | Bán kính từ chối |
| 3% | `process` | Ngưỡng cảnh báo chói |
