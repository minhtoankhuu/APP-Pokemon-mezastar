"""Tao them bien the audio tu 1 file goc cho moi nhan vat trong data/raw.

Vi moi nhan vat hien chi co 1 file audio goc, script nay dung librosa de
sinh them cac bien the (pitch shift, time stretch, them nhieu, doi am
luong, dich thoi gian) va luu thanh .wav moi trong cung thu muc nhan vat.
File goc va anh dai dien khong bi dong cham.

Chay:
    python tools/augment_audio.py --data data/raw --count 15
"""
import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
AUDIO_EXTS = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')


def is_augmented(path: Path) -> bool:
    return '_aug' in path.stem


def find_source_files(folder: Path):
    files = [p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    return [p for p in files if not is_augmented(p)]


def add_noise(audio: np.ndarray, level: float) -> np.ndarray:
    noise = np.random.randn(len(audio)).astype(np.float32)
    return audio + level * noise


def random_gain(audio: np.ndarray, low: float, high: float) -> np.ndarray:
    return audio * random.uniform(low, high)


def time_shift(audio: np.ndarray, max_fraction: float) -> np.ndarray:
    shift = int(len(audio) * random.uniform(-max_fraction, max_fraction))
    return np.roll(audio, shift)


def make_variant(audio: np.ndarray, sr: int, rng_idx: int) -> np.ndarray:
    out = audio.copy()

    n_steps = random.uniform(-2.5, 2.5)
    out = librosa.effects.pitch_shift(out, sr=sr, n_steps=n_steps)

    rate = random.uniform(0.85, 1.15)
    out = librosa.effects.time_stretch(out, rate=rate)

    out = random_gain(out, 0.7, 1.3)
    out = time_shift(out, 0.1)
    out = add_noise(out, level=random.uniform(0.001, 0.01))

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.95
    return out.astype(np.float32)


def augment_folder(folder: Path, target_count: int) -> int:
    sources = find_source_files(folder)
    if not sources:
        return 0

    existing_total = len([p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS])
    created = 0
    variant_idx = 0
    while existing_total + created < target_count:
        src = random.choice(sources)
        audio, sr = librosa.load(src, sr=SAMPLE_RATE, mono=True)
        variant = make_variant(audio, sr, variant_idx)
        out_path = folder / f'{src.stem}_aug{variant_idx}.wav'
        sf.write(out_path, variant, SAMPLE_RATE)
        created += 1
        variant_idx += 1
    return created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--count', type=int, default=15, help='So luong audio muc tieu moi nhan vat (goc + augment)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    total_created = 0
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        created = augment_folder(folder, args.count)
        total_created += created
        print(f'{folder.name}: +{created} file augment')

    print(f'Xong. Tong cong tao them {total_created} file audio.')


if __name__ == '__main__':
    main()
