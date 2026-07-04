import sys
from pathlib import Path
import shutil

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
MODELS = ROOT / 'models'
ASSETS = ROOT / 'android' / 'app' / 'src' / 'main' / 'assets'
IMAGES = ASSETS / 'images'

ASSETS.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

model_src = MODELS / 'pokemon_voice.tflite'
labels_src = MODELS / 'labels.txt'
if not model_src.exists() or not labels_src.exists():
    raise SystemExit('Chưa có model. Hãy chạy: python tools/train_audio_model.py --data data/raw --out models')

shutil.copy2(model_src, ASSETS / 'model.tflite')
shutil.copy2(labels_src, ASSETS / 'labels.txt')

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')
missing = []

for folder in sorted(DATA.iterdir()):
    if not folder.is_dir():
        continue
    candidates = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not candidates:
        missing.append(folder.name)
        continue
    image = sorted(candidates)[0]
    shutil.copy2(image, IMAGES / f'{folder.name}{image.suffix.lower()}')

if missing:
    print('Không tìm thấy ảnh cho:', ', '.join(missing))

print('Đã đồng bộ assets Android.')
