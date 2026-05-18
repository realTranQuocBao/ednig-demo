# Quickstart — chạy với Python 3.10 + GPU

Bạn đã có Python 3.10 trên máy. Mở **PowerShell** trong thư mục `D:\MASTER\BaoCaoNLP\ednig` rồi chạy lần lượt:

## 1. Tạo virtualenv với Python 3.10

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version            # phải in "Python 3.10.x"
```

Nếu PowerShell báo "running scripts is disabled" khi activate:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Rồi chạy lại `.\.venv\Scripts\Activate.ps1`.

## 2. Cài PyTorch GPU + dependencies

Chọn CUDA tương ứng với driver GPU của bạn (mở Command Prompt, gõ `nvidia-smi` để xem dòng "CUDA Version:"):

```powershell
# Nếu driver hỗ trợ CUDA 12.1+ (đa số card 30xx/40xx hiện đại):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Hoặc CUDA 11.8 (card cũ hơn, Maxwell/Pascal/Turing):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Các thư viện còn lại:
pip install numpy opencv-python pillow tqdm flask
```

## 3. Kiểm tra CUDA chạy được

```powershell
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Phải in `CUDA available: True` và tên GPU của bạn. Nếu in `False` thì driver/CUDA chưa khớp, kiểm tra lại bước 2.

## 4. Train (đầy đủ paper: 180 epochs, 512×512)

```powershell
python -m pytorch_impl.train --data .\data\lol_dataset
```

Tham số mặc định đã khớp paper. Checkpoint lưu vào `weights\g\generator_ep{NNN}.pt` và `weights\ednig_generator_latest.pt` mỗi 5 epoch.

Nếu muốn theo dõi GPU đang dùng bao nhiêu memory, mở terminal mới:

```powershell
nvidia-smi -l 2
```

Thời gian dự kiến trên một số GPU:
- RTX 4090 / 4080:  ~3-4 giờ
- RTX 3070 / 3080:  ~6-8 giờ
- RTX 3060 / 2080:  ~8-12 giờ
- GTX 1660 / 1080:  ~15-20 giờ

## 5. Chạy web app

Mở terminal mới (giữ training chạy nền nếu muốn) hoặc chạy sau khi train xong:

```powershell
.\.venv\Scripts\Activate.ps1
python webapp\app.py
```

Mở browser tại http://127.0.0.1:5000.

Web app tự load `weights\ednig_generator_latest.pt` khi nó xuất hiện. Badge trên header sẽ chuyển từ "Fallback" sang "Model đã train".

## Tóm tắt một dòng

```powershell
py -3.10 -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 ; pip install numpy opencv-python pillow tqdm flask ; python -m pytorch_impl.train --data .\data\lol_dataset
```
