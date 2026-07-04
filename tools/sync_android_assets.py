from pathlib import Path
import shutil

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

for folder in sorted(DATA.iterdir()):
    if not folder.is_dir():
        continue
    image = None
    for name in ('image.png', 'image.jpg', 'image.jpeg', 'image.webp'):
        candidate = folder / name
        if candidate.exists():
            image = candidate
            break
    if image:
        shutil.copy2(image, IMAGES / f'{folder.name}{image.suffix.lower()}')

print('Đã đồng bộ assets Android.')
