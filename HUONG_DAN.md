# EDNIG — Hướng dẫn chạy lại training và sử dụng web app

Đây là phiên bản port sang **PyTorch hiện đại** của bài báo
*"Low-Light Enhancement via Encoder-Decoder Network with Illumination Guidance"*
(Tran et al., ICCCE 2025), kèm một **web app Flask** để kéo–thả ảnh thiếu sáng
và tải về ảnh đã được tăng cường ở chất lượng cao.

## 1. Cấu trúc thư mục

```
ednig/
├── pytorch_impl/          # Implementation PyTorch (đã port từ Keras 2.1.3)
│   ├── bcp.py             #   Bright Channel Prior (illumination map)
│   ├── model.py           #   Generator (U-Net + SPP + Swish) & Discriminator
│   ├── losses.py          #   L2 + Perceptual (VGG16) + Wasserstein
│   ├── dataset.py         #   LOL dataset loader (paired low/high)
│   ├── train.py           #   Vòng huấn luyện GAN
│   └── inference.py       #   Inference + classical fallback
├── webapp/                # Web app Flask
│   ├── app.py
│   ├── templates/index.html
│   └── static/{style.css, script.js}
├── data/lol_dataset/      # Bộ dữ liệu LOL có sẵn (485 train + 15 eval)
├── weights/               # Nơi lưu checkpoint sau khi train
├── requirements_pytorch.txt
├── setup_and_run.sh       # quick start cho Linux/macOS
└── setup_and_run.bat      # quick start cho Windows
```

## 2. Cài đặt môi trường

Khuyến nghị Python 3.10+. Cài torch theo CUDA của máy bạn:

**Linux/macOS:**

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh install_gpu     # nếu có GPU NVIDIA + CUDA 12.1
# hoặc
./setup_and_run.sh install         # CPU only
```

**Windows (PowerShell hoặc CMD):**

```bat
setup_and_run.bat install_gpu
:: hoặc
setup_and_run.bat install
```

Hoặc tự cài thủ công:

```bash
pip install -r requirements_pytorch.txt
# rồi cài torch theo CUDA của bạn, ví dụ:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. Train lại model trên GPU

LOL dataset đã có sẵn trong `data/lol_dataset/`.

```bash
python -m pytorch_impl.train \
    --data ./data/lol_dataset \
    --img-size 512 \
    --epochs 180 \
    --batch-size 1 \
    --critic-updates 5 \
    --save-dir ./weights
```

Tham số mặc định lặp lại theo paper. Checkpoint được lưu mỗi 5 epoch tại
`weights/g/generator_ep{NNN}.pt`, và phiên bản mới nhất luôn được copy thành
`weights/ednig_generator_latest.pt` — đây cũng là file mà web app sẽ tự load.

Resume training:

```bash
python -m pytorch_impl.train \
    --resume-g weights/g/generator_ep050.pt \
    --resume-d weights/d/discriminator_ep050.pt
```

**Lưu ý hiệu năng:** Trên RTX 3060 / 3070, 180 epochs với ảnh 512×512 mất
khoảng 6–10 giờ. Nếu cần demo nhanh hãy thử
`--epochs 30 --img-size 256 --num-workers 4`.

## 4. Chạy web app

```bash
# Linux/macOS
./setup_and_run.sh web

# Windows
setup_and_run.bat web

# hoặc trực tiếp
python webapp/app.py
```

Mở trình duyệt tại **http://127.0.0.1:5000**.

### Tính năng

- **Kéo–thả** hoặc bấm để chọn ảnh thiếu sáng (PNG/JPG/WEBP/BMP, tối đa 25 MB).
- **Thanh trượt so sánh trước/sau** ngay trên ảnh.
- **Hai nút tải xuống**: PNG (lossless, chất lượng cao nhất) và JPG (95).
- Output giữ **độ phân giải gốc của ảnh đầu vào** — model chạy ở 512×512,
  sau đó được resize ngược về kích thước ban đầu bằng Lanczos.
- **Fallback thông minh**: nếu chưa có file `weights/ednig_generator_latest.pt`,
  web app tự dùng thuật toán "classical illumination enhancement"
  (gamma adaptive dựa trên BCP + CLAHE) để bạn vẫn xài được ngay.
  Khi đã train xong, đặt file `.pt` vào thư mục `weights/` và khởi động lại web app
  — góc trên bên phải sẽ hiện badge **"Model đã train"**.

### Biến môi trường

| Biến                  | Mặc định      | Ý nghĩa                            |
|-----------------------|---------------|------------------------------------|
| `EDNIG_HOST`          | `127.0.0.1`   | Địa chỉ lắng nghe                  |
| `EDNIG_PORT`          | `5000`        | Cổng                                |
| `EDNIG_INFER_SIZE`    | `512`         | Resize ảnh trước khi đưa vào model  |

## 5. So sánh với code gốc

| Thành phần          | Bản gốc                        | Bản port               |
|---------------------|--------------------------------|------------------------|
| Framework           | Keras 2.1.3 + TF 1.14 + CUDA 10 | PyTorch ≥ 2.1          |
| Generator           | U-Net + SPP + Swish (4 ch in)   | giữ nguyên kiến trúc   |
| Discriminator       | U-Net encoder + GAP             | giữ nguyên             |
| Losses              | L2 + VGG16-block3 + Wasserstein | giữ nguyên             |
| Augmentation        | Random crop + flip              | Resize + flip          |
| Format checkpoint   | `.h5`                           | `.pt`                  |
| Inference           | `test_on_images.py`             | `inference.py`         |
| Web app             | —                              | Flask + HTML/JS đẹp    |

## 6. Lý do port

- Keras 2.1.3 và TF 1.14 đã không còn cài được bằng pip cho Python ≥ 3.9.
- PyTorch hỗ trợ CUDA 12 và mọi GPU NVIDIA hiện đại; dễ cài đặt.
- Code PyTorch ngắn, dễ debug, không bị các vấn đề về `K.backend`, `tf.contrib`, v.v.

## 7. Trích dẫn

```bibtex
@inproceedings{tran2025low,
  title={Low-light enhancement via encoder-decoder network with illumination guidance},
  author={Tran, Le-Anh and Tran, Chung Nguyen and Nguyen, Ngoc-Luu and Dang, Nhan Cach and Carrabina, Jordi and Castells-Rufas, David and Nguyen, Minh Son},
  booktitle={2025 10th International Conference on Computer and Communication Engineering (ICCCE)},
  pages={01--06},
  year={2025},
  organization={IEEE}
}
```
