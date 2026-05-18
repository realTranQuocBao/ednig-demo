# Nhật ký công việc — EDNIG (PyTorch port + Web app)

Tài liệu này tổng kết tất cả thay đổi đã thực hiện trên project. Mục đích: bạn nắm được hiện trạng, lý do từng quyết định, các bug đã sửa và cách vận hành.

---

## 1. Tóm tắt mục tiêu

Yêu cầu ban đầu: "trong thư mục paper có bài báo gốc tiếng Anh và bài báo dịch tiếng Việt, trong thư mục data có dataset… không có model training sẵn, hãy thực hiện chạy training lại, và viết 1 ứng dụng web đơn giản để kéo thả/upload ảnh đầu vào thiếu sáng, và trả ảnh đầu ra đủ sáng… cho phép tải hình ảnh với chất lượng cao."

Đã thực hiện:

1. **Port toàn bộ code training từ Keras 2.1.3 / TF 1.14 (2019) sang PyTorch ≥ 2.1 hiện đại** — bản gốc không chạy được với Python ≥ 3.9.
2. **Bổ sung data augmentation** mà bản gốc đã tắt (`crop_sizes = []`).
3. **Sửa 2 bug nghiêm trọng trong port** (size mismatch + loss bùng nổ).
4. **Xây web app Flask + HTML/JS** với drag-drop, slider so sánh trước/sau, tải PNG lossless / JPG 95.
5. **Hiển thị thông tin tác giả + người dịch + link đọc PDF** ngay trên web.
6. **Lưu lịch sử ảnh upload + ảnh đã enhance** vào đĩa (`webapp/history/`), truy cập qua URL bí mật `/history-check` (không công khai UI).

---

## 2. Cấu trúc thư mục sau khi làm

```
D:\MASTER\BaoCaoNLP\ednig\
├── core/                     # Code Keras GỐC của paper (giữ nguyên để tham khảo)
│   ├── networks.py
│   ├── losses.py
│   ├── bcp.py
│   └── utils.py
├── pytorch_impl/             # ===== CODE PYTORCH MỚI (DÙNG CÁI NÀY) =====
│   ├── __init__.py
│   ├── bcp.py                #   Bright Channel Prior (illumination map)
│   ├── model.py              #   Generator (U-Net + SPP + Swish) + Discriminator
│   ├── losses.py             #   L2 + VGG16-perceptual + Wasserstein
│   ├── dataset.py            #   LOL dataset loader + augmentation đầy đủ
│   ├── train.py              #   GAN training loop
│   └── inference.py          #   Inference + classical fallback
├── webapp/                   # Web app Flask
│   ├── app.py                #   Routes: /, /api/enhance, /api/download, /history-check, ...
│   ├── templates/
│   │   ├── index.html        #   Trang chính (drag-drop + slider + PDF links)
│   │   └── history.html      #   Trang lịch sử ẩn (chỉ truy cập qua URL bí mật)
│   ├── static/
│   │   ├── style.css         #   Theme dark, responsive
│   │   └── script.js         #   Drag-drop + slider + upload logic
│   └── history/              #   (Không commit — chứa ảnh demo đã enhance)
├── data/lol_dataset/         # 485 train + 15 eval pairs (low/high)
├── paper/                    # PDF tiếng Anh + tiếng Việt + nguồn LaTeX
├── weights/                  # Output checkpoint (.pt) sau training
├── docs/                     # Ảnh minh hoạ paper (gốc, không sửa)
├── outputs/                  # Output paper mẫu (gốc, không sửa)
├── train.py                  # Train script CŨ Keras (KHÔNG dùng nữa)
├── test_on_images.py         # Test script CŨ Keras (KHÔNG dùng nữa)
├── requirements.txt          # CŨ Keras 2.1.3 + TF 1.14
├── requirements_pytorch.txt  # MỚI — dùng cái này
├── setup_and_run.bat         # Quick start cho Windows
├── setup_and_run.sh          # Quick start cho Linux/macOS
├── HUONG_DAN.md              # Hướng dẫn chi tiết tiếng Việt
├── QUICKSTART.md             # Quick start với Python 3.10
├── NHAT_KY_CONG_VIEC.md      # CHÍNH FILE NÀY
└── .gitignore                # Bỏ qua .venv, .pyc, weights, history
```

---

## 3. Lý do phải port sang PyTorch

Bản gốc của paper dùng:
- Keras 2.1.3 (phát hành 2017)
- TensorFlow 1.14 (phát hành 2019)
- CUDA 10.0 + cuDNN 7.6

Các phiên bản này:
- Không cài được bằng `pip install` cho Python ≥ 3.9
- Không hỗ trợ GPU NVIDIA hiện đại (Ampere/Ada/Hopper)
- Không có wheel cho Windows hiện tại

Vì vậy tôi giữ nguyên bản Keras gốc trong `core/` để tham khảo nhưng viết bản PyTorch mới tương đương trong `pytorch_impl/`. Kiến trúc model giữ NGUYÊN 100% so với paper.

---

## 4. Các bug đã sửa trong quá trình port (quan trọng — đừng quên!)

### Bug #1: Tensor size mismatch trong UpBlock (đã sửa)

**Triệu chứng:**
```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 64 but got size 65 for tensor number 1 in the list.
```

**Nguyên nhân:** Keras `Conv2D(kernel=2, padding='same')` pad bất đối xứng `(left=0, right=1, top=0, bottom=1)` để giữ nguyên spatial size. PyTorch `Conv2d(kernel=2, padding=1)` pad đối xứng → output to thêm 1 pixel. Skip connection trong U-Net concat 64×64 với 65×65 → crash.

**Fix** trong `pytorch_impl/model.py::UpBlock`:
```python
def forward(self, x):
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = F.pad(x, (0, 1, 0, 1))     # asymmetric pad giống Keras 'same'
    x = self.conv(x)                # kernel=2, padding=0
    return swish(x)
```

### Bug #2: Loss bùng nổ (đã sửa)

**Triệu chứng:** Sau ~50 step training, `d_loss` và `g_loss` vọt lên 10²² (mười chấm xíu), 10²³…

**Nguyên nhân:** Bản gốc Keras có `Dense(1, activation='sigmoid')` ở head Discriminator — output bounded `[0, 1]`. Khi áp `wasserstein_loss = mean(y_true * y_pred)` thì loss bị chặn |L| ≤ 1, gradient cũng chặn → ổn định.

Khi port sang PyTorch tôi quên sigmoid → critic output không giới hạn → mỗi bước update làm output to gấp đôi → bùng nổ theo cấp số nhân.

**Fix** trong `pytorch_impl/model.py::Discriminator`:
```python
self.head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(f5, 128),
    nn.LeakyReLU(0.1, inplace=True),
    nn.Linear(128, 1),
    nn.Sigmoid(),       # ← thêm dòng này, khớp Keras
)
```

### Bug #3: File ghi đĩa bị truncated/null-padded khi dùng Cowork Write tool

**Triệu chứng:** `app.py` và `inference.py` đôi khi bị ngắt giữa chừng, padding null bytes.

**Fix:** Ghi lại các file đó qua bash heredoc (`cat > file << EOF`) thay vì Write tool. Đã ổn ở lần ghi cuối.

---

## 5. Augmentation đã thêm (bản gốc Keras không có)

Trong `core/utils.py` của paper, `crop_sizes = []` và đoạn flip code cuối cùng bị comment out — tức là bản gốc thực ra **không có augment**, chỉ resize ảnh.

`pytorch_impl/dataset.py` của tôi áp dụng đầy đủ:

| # | Augmentation | Xác suất | Tác dụng |
|---|---|---|---|
| 1 | Random multi-scale crop | 100% | Crop vuông size `uniform(0.5–1.0) × min(H,W)` rồi resize 512 |
| 2 | Horizontal flip | 50% | Đối xứng trái-phải |
| 3 | Vertical flip | 20% | Nhẹ |
| 4 | Rotation 90/180/270° | 30% | Nhân 4× orientation |
| 5 | Photometric jitter (LOW only) | 15% | Brightness/contrast ±10%, KHÔNG áp lên high |

Lưu ý: BCP illumination map được tính LẠI sau augment để khớp với input thực model thấy.

---

## 6. Kiến trúc model (giữ nguyên paper)

### Generator: EDN-GIM (Encoder-Decoder Network with Guidance from Illumination Map)

- Input: **4 channels** = RGB (3) + BCP illumination map (1)
- U-Net 5 cấp với skip connection
- Encoder: 4 cấp ConvBlock + MaxPool (3 conv 3×3 Swish mỗi cấp)
- Bottleneck: ConvBlock + **Spatial Pyramid Pooling** (5/9/13 max-pool)
- Decoder: 4 cấp UpBlock + concat skip + ConvBlock
- Activation: **Swish** (= x · sigmoid(x))
- Output: 3 channels RGB qua **tanh** → range [-1, 1]
- Trọng số: ~865K parameters (model nhẹ)

### Discriminator (Critic)

- Input: 3 channels RGB
- 5 cấp Conv+ReLU+MaxPool
- Global Average Pool + Dense(128) + Dense(1)
- Sigmoid (**đã thêm sau bug fix**)

### Loss

- Generator loss = **100 × (perceptual_VGG16-block3 + 10 × L2)** + **1 × Wasserstein**
- Wasserstein loss = mean(y_true × D(x)) với y_true ∈ {+1, -1}
- Critic update **5 lần** mỗi 1 lần generator update

### Optimizer

- Adam(lr=1e-4, β1=0.9, β2=0.999), linear decay
- 180 epochs, batch_size=1, img_size=512×512

---

## 7. Web app — tính năng

### Trang chính `/`

- Card hiển thị thông tin **tác giả gốc** (Tran et al., ICCCE 2025) + nút mở PDF tiếng Anh
- Card hiển thị **người dịch Trần Quốc Bảo (2480101829)** + nút mở PDF tiếng Việt
- Drag-drop hoặc click chọn ảnh (PNG/JPG/WEBP/BMP, ≤25MB)
- Sau khi enhance: slider trượt so sánh trước/sau ngay trên ảnh
- Badge trạng thái: "Model đã train" / "Fallback (chưa có model)"
- 2 nút tải xuống: **PNG lossless** (chất lượng cao nhất) và **JPG 95**
- Output giữ **độ phân giải gốc** — model chạy ở 512×512 rồi resize ngược bằng Lanczos

### Trang lịch sử ẩn `/history-check`

- KHÔNG có link trên UI chính (chỉ vào được khi gõ URL)
- Grid hiển thị tất cả ảnh đã enhance + metadata (timestamp, filename, dimensions, mode, brightness)
- Nút tải xuống / xoá từng run
- API JSON: `/history-check/api`

### Cơ chế lưu trữ

Mỗi lần enhance lưu vào `webapp/history/<YYYYMMDD_HHMMSS>_<sid>/`:
- `input.png` — ảnh gốc lossless
- `output.png` — ảnh đã enhance lossless, full resolution
- `meta.json` — original filename, mode, kích thước, brightness, ms, timestamp

Tự động giới hạn `MAX_HISTORY=200` runs gần nhất (cấu hình qua env `EDNIG_MAX_HISTORY`).

### Fallback thông minh

Khi chưa có file `weights/ednig_generator_latest.pt`, web app dùng "classical illumination enhancement" (BCP + adaptive gamma + CLAHE) thay vì báo lỗi. Người dùng vẫn enhance được ảnh ngay từ đầu, đến khi train xong và restart web app thì badge chuyển sang "Model đã train".

---

## 8. Cách chạy (tóm tắt)

### Setup môi trường (1 lần)

```powershell
cd D:\MASTER\BaoCaoNLP\ednig
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài torch GPU theo CUDA driver (driver 12.6 → cu121 wheel)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python pillow tqdm flask
```

### Training full theo paper

```powershell
# Xoá weights cũ trước
Remove-Item -Recurse -Force .\weights\g\, .\weights\d\, .\weights\ednig_generator_latest.pt, .\pytorch_impl\__pycache__\ -ErrorAction SilentlyContinue

# Smoke test 2 epoch — verify loss ổn định
python -m pytorch_impl.train --data .\data\lol_dataset --epochs 2 --save-every 1

# Nếu OK (d_loss và adv trong [-1, 1]) → xoá và train full
Remove-Item -Recurse -Force .\weights\g\, .\weights\d\, .\weights\ednig_generator_latest.pt -ErrorAction SilentlyContinue
python -m pytorch_impl.train --data .\data\lol_dataset
```

Với RTX 3060 ~85s/epoch → 180 epochs ≈ **4-5 giờ**.

### Chạy web app

```powershell
python webapp\app.py
```

Mở http://127.0.0.1:5000.

Web app load `weights/ednig_generator_latest.pt` lúc khởi động. Sau khi train xong cần **restart** web app để load model mới.

---

## 9. So sánh với code gốc

| Thành phần | Bản gốc Keras | Bản port PyTorch |
|---|---|---|
| Framework | Keras 2.1.3 + TF 1.14 | PyTorch ≥ 2.1 |
| Python | 2.7 / 3.6 | 3.10+ |
| CUDA | 10.0 | 11.8 / 12.1 / 12.4 |
| Generator | U-Net + SPP + Swish 4ch | **giữ nguyên** kiến trúc |
| Discriminator | U-Net encoder + sigmoid head | **giữ nguyên** (đã sửa thiếu sigmoid) |
| Losses | L2 + VGG16 + Wasserstein | **giữ nguyên** trọng số |
| Data augment | Tắt (`crop_sizes=[]`) | **5 phép augment** chuẩn |
| Format checkpoint | `.h5` | `.pt` |
| Inference script | `test_on_images.py` | `inference.py` + Flask |
| Web app | Không có | **Có** (Flask + HTML/JS) |
| Lưu lịch sử | Không có | **Có** (folder + UI ẩn) |
| Hiển thị tác giả/người dịch | Không có | **Có** (card + link PDF) |

---

## 10. Roadmap khuyến nghị

Việc bạn cần làm:
1. ✅ Cài Python 3.10 + venv + torch GPU
2. ✅ Smoke test 2 epoch — verify loss không bùng nổ
3. 🔄 Train full 180 epochs (~4-5 giờ trên RTX 3060)
4. 🔄 Restart web app → badge chuyển "Model đã train"
5. 🔄 Demo: upload ảnh thiếu sáng, so sánh trước/sau, tải xuống

Có thể làm thêm nếu muốn:
- Chạy `inference.py` standalone trên 1 thư mục ảnh để export ảnh hàng loạt
- Test trên LOL eval15 để tính PSNR/SSIM so sánh với paper
- Tinh chỉnh augmentation strength qua các tham số `crop_min`, `flip_prob`, ...
- Đóng gói executable bằng PyInstaller cho người không cần Python

---

## 11. File quan trọng cần biết

| File | Vai trò |
|---|---|
| `pytorch_impl/model.py` | Generator + Discriminator (kiến trúc paper) |
| `pytorch_impl/train.py` | Vòng GAN training |
| `pytorch_impl/inference.py` | Inference + classical fallback |
| `pytorch_impl/dataset.py` | LOL loader + augmentation |
| `pytorch_impl/losses.py` | VGG16 perceptual + L2 + Wasserstein |
| `pytorch_impl/bcp.py` | Bright Channel Prior (illumination map) |
| `webapp/app.py` | Flask routes + lưu lịch sử |
| `webapp/templates/index.html` | UI trang chính |
| `webapp/templates/history.html` | UI lịch sử ẩn |
| `webapp/static/style.css`, `script.js` | Theme + drag-drop logic |
| `HUONG_DAN.md` | Hướng dẫn dài chi tiết |
| `QUICKSTART.md` | Setup nhanh Python 3.10 + GPU |

---

## 12. Trạng thái cuối

- ✅ Code PyTorch chạy được trên CPU + GPU
- ✅ Augmentation đầy đủ
- ✅ 2 bug nghiêm trọng đã fix (size mismatch + loss bùng nổ)
- ✅ Web app hoạt động, drag-drop, so sánh, tải PNG/JPG chất lượng cao
- ✅ Lưu lịch sử ảnh demo vào đĩa
- ✅ Hiển thị thông tin tác giả + người dịch + link PDF
- ⏳ **Chờ bạn train full 180 epochs trên RTX 3060** để có model thật
- ⏳ Sau khi train xong → restart web app để load `ednig_generator_latest.pt`

---

*Tài liệu cập nhật lần cuối khi vừa thêm augmentation và fix sigmoid bug ở Discriminator.*
