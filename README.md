# App APK nhận dạng tiếng nhân vật và trả hình

Project này gồm 2 phần:

- `tools/`: script Python để chuẩn bị dữ liệu và train model TensorFlow Lite.
- `android/`: project Android Studio để build APK.

## 1. Chuẩn bị dữ liệu

Mỗi nhân vật đặt trong một thư mục riêng:

```text
data/raw/pikachu/
  image.png
  001.wav
  002.wav
  003.wav

data/raw/charmander/
  image.png
  001.wav
  002.wav
```

Quy ước:

- Tên thư mục chính là nhãn nhân vật, nên viết không dấu, không khoảng trắng. Ví dụ: `pikachu`, `bulbasaur`, `charizard`.
- Mỗi nhân vật nên có ít nhất 10 mẫu âm thanh, tốt hơn là 20-50 mẫu.
- File ảnh đại diện nên đặt tên `image.png` hoặc `image.jpg`.
- Audio nên là `.wav`, dài khoảng 1-3 giây. Script sẽ tự chuyển về 16 kHz mono khi train nếu đọc được.

Với 72 nhân vật, cấu trúc sẽ giống:

```text
data/raw/nhan_vat_001/
data/raw/nhan_vat_002/
...
data/raw/nhan_vat_072/
```

## 2. Cài môi trường train

Cài Python 3.10+ rồi chạy trong thư mục project:

```powershell
pip install -r requirements.txt
```

## 3. Train model

```powershell
python tools/train_audio_model.py --data data/raw --out models --duration 2.0
```

Sau khi train xong sẽ có:

```text
models/pokemon_voice.tflite
models/labels.txt
models/training_report.json
```

## 4. Chép model và ảnh vào Android

```powershell
python tools/sync_android_assets.py
```

Script sẽ chép:

- `models/pokemon_voice.tflite` vào `android/app/src/main/assets/model.tflite`
- `models/labels.txt` vào `android/app/src/main/assets/labels.txt`
- ảnh từng nhân vật vào `android/app/src/main/assets/images/`

## 5. Build APK

Mở thư mục `android/` bằng Android Studio, đợi Gradle sync, rồi chọn:

```text
Build > Build Bundle(s) / APK(s) > Build APK(s)
```

## Ghi chú quan trọng

Bản đầu tiên dùng model nhận dạng waveform 2 giây. Khi bấm nút nghe, app ghi âm 2 giây rồi dự đoán nhân vật. Nếu kết quả sai nhiều, cần thêm audio mẫu cho các nhân vật dễ nhầm nhau.

## 6. Nếu bạn có một video/audio tổng hợp

Mình đã thêm video mẫu vào:

```text
data/source/7976531600508.mp4
```

Bạn điền mốc thời gian từng nhân vật trong:

```text
data/source/segments.csv
```

Ví dụ:

```csv
label,start,end
pikachu,00:00:01.000,00:00:03.000
charizard,00:00:04.000,00:00:06.000
```

Sau đó cài FFmpeg, rồi chạy:

```powershell
python tools/split_source_video.py --video data/source/7976531600508.mp4 --segments data/source/segments.csv --out data/raw
```

Script sẽ tạo thư mục theo từng nhân vật, tách audio `.wav`, và lấy ảnh đại diện từ khung hình giữa đoạn đó.

## 7. Tăng cường dữ liệu khi mỗi nhân vật chỉ có 1 file audio

Nếu mỗi thư mục trong `data/raw` chỉ có 1 file audio gốc (không đủ để train, nên có ít nhất 10-15 mẫu/nhân vật), chạy:

```powershell
python tools/augment_audio.py --data data/raw --count 15
```

Script sẽ giữ nguyên file gốc và ảnh, chỉ sinh thêm các file `.wav` biến thể (đổi cao độ, tốc độ, âm lượng, thêm nhiễu nhẹ) cho tới khi mỗi nhân vật có đủ `--count` file audio. Chạy lại lệnh với `--count` lớn hơn nếu muốn thêm mẫu.

Lưu ý: dữ liệu augment giúp pipeline chạy được nhưng độ chính xác thực tế vẫn kém hơn nhiều so với có audio thật đa dạng. Khi có điều kiện, nên thay dần các file `_aug` bằng audio ghi âm/tách thật.

## 8. Train trên máy khác (máy yếu thì nên train ở máy khác mạnh hơn)

Máy hiện tại có thể không đủ mạnh hoặc không có GPU tương thích (trên Windows, TensorFlow từ bản 2.11 trở đi **không** dùng được GPU NVIDIA trực tiếp nữa, phải qua WSL2). Nếu muốn train ở máy khác (PC khác, laptop mạnh hơn, hoặc máy có GPU), làm theo các bước sau:

### 8.1. Copy project sang máy khác

Chỉ cần copy các thư mục/file sau (không cần copy `models/`, `__pycache__`, hay `android/app/build`):

```text
data/            (đặc biệt là data/raw)
tools/
requirements.txt
```

Có thể nén thành file `.zip` rồi copy qua USB, hoặc đẩy lên GitHub/Google Drive rồi tải về máy kia.

### 8.2. Cài môi trường ở máy khác

Cài Python 3.10+ ở máy đó, rồi trong thư mục project chạy:

```powershell
python -m pip install -r requirements.txt
```

Nếu máy đó có GPU NVIDIA và chạy Linux (hoặc WSL2 trên Windows), có thể cài thêm bản có hỗ trợ CUDA để train nhanh hơn nhiều:

```bash
pip install "tensorflow[and-cuda]"
```

### 8.3. Train như bình thường ở máy đó

```powershell
python tools/augment_audio.py --data data/raw --count 15
python tools/train_audio_model.py --data data/raw --out models --duration 2.0
```

### 8.4. Copy kết quả train về lại máy này

Sau khi train xong ở máy kia, chỉ cần copy 3 file trong thư mục `models/` về đúng vị trí `models/` của project trên máy này:

```text
models/pokemon_voice.tflite
models/labels.txt
models/training_report.json
```

Sau đó chạy tiếp như bình thường ở máy này:

```powershell
python tools/sync_android_assets.py
```

rồi build APK bằng Android Studio như bước 5.

### 8.5. Không có máy nào mạnh hơn? Dùng Google Colab (miễn phí, có GPU)

1. Vào [colab.research.google.com](https://colab.research.google.com), tạo notebook mới, chọn **Runtime > Change runtime type > GPU**.
2. Upload thư mục `data/raw` và `tools/` lên Colab (kéo thả vào panel Files, hoặc zip rồi `!unzip`).
3. Chạy trong 1 cell:

```python
!pip install -r requirements.txt
!python tools/augment_audio.py --data data/raw --count 15
!python tools/train_audio_model.py --data data/raw --out models --duration 2.0
```

4. Tải 3 file trong `models/` về máy (chuột phải > Download trong panel Files), rồi copy vào `models/` của project và làm tiếp bước 8.4.
