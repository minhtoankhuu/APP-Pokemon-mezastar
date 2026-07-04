import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 16000


def load_audio(path: Path, samples: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < samples:
        audio = np.pad(audio, (0, samples - len(audio)))
    elif len(audio) > samples:
        start = random.randint(0, len(audio) - samples)
        audio = audio[start:start + samples]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def collect_dataset(data_dir: Path, duration: float):
    samples = int(SAMPLE_RATE * duration)
    labels = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if len(labels) < 2:
        raise SystemExit('Cần ít nhất 2 nhân vật để train model.')

    x, y = [], []
    counts = {}
    for idx, label in enumerate(labels):
        folder = data_dir / label
        audio_files = []
        for ext in ('*.wav', '*.mp3', '*.m4a', '*.ogg', '*.flac'):
            audio_files.extend(folder.glob(ext))
        counts[label] = len(audio_files)
        if not audio_files:
            print(f'Bỏ qua {label}: không có audio.')
            continue
        for file in audio_files:
            try:
                x.append(load_audio(file, samples))
                y.append(idx)
            except Exception as exc:
                print(f'Không đọc được {file}: {exc}')

    if not x:
        raise SystemExit('Không tìm thấy audio hợp lệ.')

    return np.stack(x), np.array(y, dtype=np.int64), labels, counts


def build_model(input_samples: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_samples, 1), name='audio')
    x = tf.keras.layers.Conv1D(16, 9, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(32, 9, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 9, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(96, 9, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='character')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--out', default='models')
    parser.add_argument('--duration', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_samples = int(SAMPLE_RATE * args.duration)
    x, y, labels, counts = collect_dataset(data_dir, args.duration)
    x = x[..., np.newaxis]

    stratify = y if min(np.bincount(y, minlength=len(labels))) >= 2 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = build_model(input_samples, len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=16,
        callbacks=callbacks,
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    (out_dir / 'pokemon_voice.tflite').write_bytes(tflite_model)
    (out_dir / 'labels.txt').write_text('\n'.join(labels), encoding='utf-8')

    report = {
        'sample_rate': SAMPLE_RATE,
        'duration_seconds': args.duration,
        'input_samples': input_samples,
        'labels': labels,
        'audio_counts': counts,
        'last_train_accuracy': float(history.history['accuracy'][-1]),
        'last_val_accuracy': float(history.history['val_accuracy'][-1]),
    }
    (out_dir / 'training_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Đã tạo model:', out_dir / 'pokemon_voice.tflite')


if __name__ == '__main__':
    main()
